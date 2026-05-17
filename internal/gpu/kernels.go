package gpu

// evaluateKernelSource contains all OpenCL kernels used by the engine.
//
// evaluate_candidates_v3:
//   For each candidate (cx, cy, rx, ry, thetaDeg, alpha) it computes the
//   geometrize-style "optimal color" (alpha-weighted mean of the target
//   minus current contribution inside the shape) and then the resulting
//   delta error if that shape were to be drawn over the current canvas.
//   Output per candidate: 4 floats { score, R, G, B }. Score is
//   (newError - oldError) summed over inside pixels (negative = better).
//   Pixels outside the opaque mask are simply skipped (they neither score
//   nor invalidate the candidate). A candidate that does not cover any
//   opaque pixel is rejected with +Inf so the engine ignores it.
//
// apply_candidate_v2:
//   Blends a chosen shape into the current canvas. Operates on a tight
//   bounding box and skips transparent pixels.
//
// compute_error_grid:
//   Buckets the per-pixel squared error of (target - current) into a
//   downsampled gridW x gridH buffer. The host side reads it back and uses
//   it to bias random candidate placement towards high-error regions.
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

    // Pass 1: accumulate target / current means inside the ellipse.
    float sumTR = 0.0f, sumTG = 0.0f, sumTB = 0.0f;
    float sumCR = 0.0f, sumCG = 0.0f, sumCB = 0.0f;
    int count = 0;

    for (int y = yMin; y <= yMax; ++y) {
        int row = y * width;
        float dy = ((float)y + 0.5f) - cy;
        for (int x = xMin; x <= xMax; ++x) {
            int p = row + x;
            if (opaqueMask[p] == 0) {
                continue;
            }
            float dx = ((float)x + 0.5f) - cx;
            float xr = dx * cosT + dy * sinT;
            float yr = -dx * sinT + dy * cosT;
            if (xr * xr * invRX2 + yr * yr * invRY2 > 1.0f) {
                continue;
            }

            float4 t = target[p];
            float4 s = current[p];
            sumTR += t.x; sumTG += t.y; sumTB += t.z;
            sumCR += s.x; sumCG += s.y; sumCB += s.z;
            count++;
        }
    }

    if (count == 0) {
        // Shape doesn't cover any opaque pixel; reject hard.
        results[gid * 4 + 0] = 3.402823466e+38f;
        results[gid * 4 + 1] = 0.0f;
        results[gid * 4 + 2] = 0.0f;
        results[gid * 4 + 3] = 0.0f;
        return;
    }

    float invCount = 1.0f / (float)count;
    float meanTR = sumTR * invCount;
    float meanTG = sumTG * invCount;
    float meanTB = sumTB * invCount;
    float meanCR = sumCR * invCount;
    float meanCG = sumCG * invCount;
    float meanCB = sumCB * invCount;

    // Optimal color: target = current * (1 - a) + color * a
    //   => color = (target - current * (1 - a)) / a   (averaged inside).
    float invA = 1.0f - ca;
    float oR = clamp((meanTR - meanCR * invA) / ca, 0.0f, 1.0f);
    float oG = clamp((meanTG - meanCG * invA) / ca, 0.0f, 1.0f);
    float oB = clamp((meanTB - meanCB * invA) / ca, 0.0f, 1.0f);

    // Pass 2: score with the optimal color.
    float totalDelta = 0.0f;
    for (int y = yMin; y <= yMax; ++y) {
        int row = y * width;
        float dy = ((float)y + 0.5f) - cy;
        for (int x = xMin; x <= xMax; ++x) {
            int p = row + x;
            if (opaqueMask[p] == 0) {
                continue;
            }
            float dx = ((float)x + 0.5f) - cx;
            float xr = dx * cosT + dy * sinT;
            float yr = -dx * sinT + dy * cosT;
            if (xr * xr * invRX2 + yr * yr * invRY2 > 1.0f) {
                continue;
            }

            float4 t = target[p];
            float4 s = current[p];

            float dr0 = t.x - s.x;
            float dg0 = t.y - s.y;
            float db0 = t.z - s.z;
            float da0 = t.w - s.w;
            float oldErr = dr0 * dr0 + dg0 * dg0 + db0 * db0 + da0 * da0;

            float nR = s.x * invA + oR * ca;
            float nG = s.y * invA + oG * ca;
            float nB = s.z * invA + oB * ca;
            float nA = s.w * invA + ca;

            float dr1 = t.x - nR;
            float dg1 = t.y - nG;
            float db1 = t.z - nB;
            float da1 = t.w - nA;
            float newErr = dr1 * dr1 + dg1 * dg1 + db1 * db1 + da1 * da1;

            totalDelta += (newErr - oldErr);
        }
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
