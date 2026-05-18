package gpu

// evaluateKernelSource contains all OpenCL kernels used by the engine.
//
// evaluate_candidates_v3 (single-pass, analytic):
//   For each candidate (cx, cy, rx, ry, thetaDeg, alpha) it computes the
//   geometrize-style "optimal color" (alpha-weighted mean of the target
//   minus current contribution inside the shape) and the resulting delta
//   error if that shape were drawn over the current canvas.
//
//   Crucially this is done in a SINGLE pass over the bounding box. We
//   accumulate per-channel statistics
//     N    = inside pixel count
//     Σs   = sum of current
//     Σt   = sum of target
//     Σs²  = sum of current squared
//     Σst  = sum of current * target
//   and then derive both the optimal RGB color AND the analytic delta
//   error in O(1) post-loop work via the identity
//
//     Σ ΔErr = α²·[N·c² − 2c·Σs + Σs²] − 2α·[c·Σt − c·Σs − Σst + Σs²]
//
//   which holds for any constant c summed over the ellipse. The alpha
//   channel uses the same identity with c=1 (because out_a blends
//   towards 1 weighted by α, not towards a free color parameter).
//
//   Skipping the second bbox traversal halves the work-item runtime at
//   the cost of ~17 extra register-resident accumulators per work-item.
//
//   Output per candidate: 4 floats { score, R, G, B }. Score is summed
//   over inside pixels (negative = better). Opaque pixels feed the
//   optimal-color statistics; transparent pixels inside the ellipse are
//   counted separately as Nt. Because the FH6 in-game renderer ignores
//   our alpha mask and paints the *full* ellipse, any shape that
//   extends into the transparent region produces a visible halo against
//   whatever body colour the user picked. We therefore HARD REJECT
//   candidates whose overhang exceeds a small tolerance (Nt*20 > N,
//   i.e. >~5% of inside pixels are transparent). The tolerance keeps
//   anti-aliased silhouette edges usable.
//
// apply_candidate_v2:
//   Blends a chosen shape into the current canvas. Operates on a tight
//   bounding box and skips transparent pixels.
//
// compute_error_grid:
//   Buckets the per-pixel squared error of (target - current) into a
//   downsampled gridW x gridH buffer. The host side reads it back and
//   uses it to bias random candidate placement towards high-error
//   regions.
const evaluateKernelSource = `
__kernel void evaluate_candidates_v3(
    __global const float4* target,
    __global const float4* current,
    __global const uchar* opaqueMask,
    __global const float* candidates,
    __global float* results,
    const int width,
    const int height
) {
    int gid = get_global_id(0);

    int base = gid * 6;
    float cx = candidates[base + 0];
    float cy = candidates[base + 1];
    float rx = fmax(candidates[base + 2], 1.0f);
    float ry = fmax(candidates[base + 3], 1.0f);
    float thetaDeg = candidates[base + 4];
    float ca = clamp(candidates[base + 5], 1e-3f, 1.0f);

    float theta = thetaDeg * 0.01745329251994329577f;
    float cosT = cos(theta);
    float sinT = sin(theta);
    float invRX2 = 1.0f / (rx * rx);
    float invRY2 = 1.0f / (ry * ry);

    float rx2 = rx * rx;
    float ry2 = ry * ry;
    float cos2 = cosT * cosT;
    float sin2 = sinT * sinT;
    float ex = sqrt(rx2 * cos2 + ry2 * sin2);
    float ey = sqrt(rx2 * sin2 + ry2 * cos2);

    int xMin = (int)floor(cx - ex - 1.0f);
    int xMax = (int)ceil(cx + ex + 1.0f);
    int yMin = (int)floor(cy - ey - 1.0f);
    int yMax = (int)ceil(cy + ey + 1.0f);

    xMin = max(0, xMin);
    yMin = max(0, yMin);
    xMax = min(width - 1, xMax);
    yMax = min(height - 1, yMax);

    // Per-channel statistics. 17 floats live in registers across the
    // bbox iteration; modern GPUs have plenty of headroom for this.
    int N = 0;   // opaque pixels inside the ellipse
    int Nt = 0;  // transparent pixels inside the ellipse (penalty bucket)
    float sTR = 0.0f, sTG = 0.0f, sTB = 0.0f, sTA = 0.0f;       // Σ target
    float sCR = 0.0f, sCG = 0.0f, sCB = 0.0f, sCA = 0.0f;       // Σ current
    float sCR2 = 0.0f, sCG2 = 0.0f, sCB2 = 0.0f, sCA2 = 0.0f;   // Σ current²
    float sTCR = 0.0f, sTCG = 0.0f, sTCB = 0.0f, sTCA = 0.0f;   // Σ target·current

    for (int y = yMin; y <= yMax; ++y) {
        int row = y * width;
        float dy = ((float)y + 0.5f) - cy;
        for (int x = xMin; x <= xMax; ++x) {
            float dx = ((float)x + 0.5f) - cx;
            float xr = dx * cosT + dy * sinT;
            float yr = -dx * sinT + dy * cosT;
            if (xr * xr * invRX2 + yr * yr * invRY2 > 1.0f) {
                continue;
            }
            int p = row + x;
            if (opaqueMask[p] == 0) {
                Nt++;
                continue;
            }

            float4 t = target[p];
            float4 s = current[p];

            sTR += t.x; sTG += t.y; sTB += t.z; sTA += t.w;
            sCR += s.x; sCG += s.y; sCB += s.z; sCA += s.w;
            sCR2 += s.x * s.x;
            sCG2 += s.y * s.y;
            sCB2 += s.z * s.z;
            sCA2 += s.w * s.w;
            sTCR += t.x * s.x;
            sTCG += t.y * s.y;
            sTCB += t.z * s.z;
            sTCA += t.w * s.w;
            N++;
        }
    }

    // Hard reject:
    //   - shapes covering no opaque pixel at all
    //   - shapes whose transparent overhang exceeds ~5% of inside area
    // The 20*Nt > N test is equivalent to Nt/(N+Nt) > ~4.76%.
    // TODO: canary build test 1% inside area to anti overhang
    if (N == 0 || Nt * 100 > N) {
        results[gid * 4 + 0] = 3.402823466e+38f;
        results[gid * 4 + 1] = 0.0f;
        results[gid * 4 + 2] = 0.0f;
        results[gid * 4 + 3] = 0.0f;
        return;
    }

    float Nf = (float)N;
    float invN = 1.0f / Nf;
    float invA = 1.0f - ca;

    // Optimal RGB color: c = (mean(t) − mean(s)·(1−α)) / α  (clamped).
    // The clamped color is what will actually be rendered, so we plug
    // that same value into the analytic ΔErr formula below — the
    // identity holds for any constant c, not just the unclamped optimum.
    float oR = clamp((sTR * invN - (sCR * invN) * invA) / ca, 0.0f, 1.0f);
    float oG = clamp((sTG * invN - (sCG * invN) * invA) / ca, 0.0f, 1.0f);
    float oB = clamp((sTB * invN - (sCB * invN) * invA) / ca, 0.0f, 1.0f);

    // Σ ΔErr = α²·[N·c² − 2c·Σs + Σs²] − 2α·[c·Σt − c·Σs − Σst + Σs²]
    //
    // RGB channels use the optimised c above. The alpha channel blends
    // toward 1 (out_a = s_a·(1−α) + α), so we substitute c = 1 below.
    float a2 = ca * ca;
    float two_a = 2.0f * ca;

    float dR = a2 * (Nf*oR*oR - 2.0f*oR*sCR + sCR2)
             - two_a * (oR*sTR - oR*sCR - sTCR + sCR2);
    float dG = a2 * (Nf*oG*oG - 2.0f*oG*sCG + sCG2)
             - two_a * (oG*sTG - oG*sCG - sTCG + sCG2);
    float dB = a2 * (Nf*oB*oB - 2.0f*oB*sCB + sCB2)
             - two_a * (oB*sTB - oB*sCB - sTCB + sCB2);
    float dA = a2 * (Nf - 2.0f*sCA + sCA2)
             - two_a * (sTA - sCA - sTCA + sCA2);

    float totalDelta = dR + dG + dB + dA;

    // Soft penalty within the tolerance budget: each remaining
    // transparent pixel still adds α²·(1+oR²+oG²+oB²) so that the
    // optimiser prefers zero overhang over allowed overhang when the
    // opaque-side error is otherwise tied.
    if (Nt > 0) {
        float penalty = a2 * ((float)Nt) * (oR*oR + oG*oG + oB*oB + 1.0f);
        totalDelta += penalty;
    }

    results[gid * 4 + 0] = totalDelta;
    results[gid * 4 + 1] = oR;
    results[gid * 4 + 2] = oG;
    results[gid * 4 + 3] = oB;
}

__kernel void apply_candidate_v2(
    __global float4* current,
    __global const uchar* opaqueMask,
    const int width,
    const int height,
    const int xMin,
    const int yMin,
    const int xMax,
    const int yMax,
    const float cx,
    const float cy,
    const float rxRaw,
    const float ryRaw,
    const float thetaDeg,
    const float cr,
    const float cg,
    const float cb,
    const float ca
) {
    int lx = get_global_id(0);
    int ly = get_global_id(1);
    int bw = xMax - xMin + 1;
    int bh = yMax - yMin + 1;
    if (lx >= bw || ly >= bh) {
        return;
    }
    int x = xMin + lx;
    int y = yMin + ly;
    int p = y * width + x;
    if (opaqueMask[p] == 0) {
        return;
    }

    float rx = fmax(rxRaw, 1.0f);
    float ry = fmax(ryRaw, 1.0f);
    float theta = thetaDeg * 0.01745329251994329577f;
    float cosT = cos(theta);
    float sinT = sin(theta);
    float invRX2 = 1.0f / (rx * rx);
    float invRY2 = 1.0f / (ry * ry);

    float dx = ((float)x + 0.5f) - cx;
    float dy = ((float)y + 0.5f) - cy;
    float xr = dx * cosT + dy * sinT;
    float yr = -dx * sinT + dy * cosT;
    if (xr * xr * invRX2 + yr * yr * invRY2 > 1.0f) {
        return;
    }

    float4 src = current[p];
    float alpha = clamp(ca, 0.0f, 1.0f);
    float invA = 1.0f - alpha;
    src.x = src.x * invA + clamp(cr, 0.0f, 1.0f) * alpha;
    src.y = src.y * invA + clamp(cg, 0.0f, 1.0f) * alpha;
    src.z = src.z * invA + clamp(cb, 0.0f, 1.0f) * alpha;
    src.w = src.w * invA + alpha;
    current[p] = src;
}

__kernel void compute_error_grid(
    __global const float4* target,
    __global const float4* current,
    __global const uchar* opaqueMask,
    __global float* gridOut,
    const int width,
    const int height,
    const int gridW,
    const int gridH
) {
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    if (gx >= gridW || gy >= gridH) {
        return;
    }

    int x0 = (int)(((long)gx * (long)width) / (long)gridW);
    int x1 = (int)(((long)(gx + 1) * (long)width) / (long)gridW);
    int y0 = (int)(((long)gy * (long)height) / (long)gridH);
    int y1 = (int)(((long)(gy + 1) * (long)height) / (long)gridH);

    float sum = 0.0f;
    for (int y = y0; y < y1; ++y) {
        int row = y * width;
        for (int x = x0; x < x1; ++x) {
            int p = row + x;
            if (opaqueMask[p] == 0) {
                continue;
            }
            float4 t = target[p];
            float4 s = current[p];
            float dr = t.x - s.x;
            float dg = t.y - s.y;
            float db = t.z - s.z;
            float da = t.w - s.w;
            sum += dr * dr + dg * dg + db * db + da * da;
        }
    }
    gridOut[gy * gridW + gx] = sum;
}
`
