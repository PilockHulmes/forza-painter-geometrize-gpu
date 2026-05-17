package engine

import (
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"time"

	"forza-painter-geometrize-go/internal/config"
	"forza-painter-geometrize-go/internal/gpu"
	"forza-painter-geometrize-go/internal/imageutil"
	"forza-painter-geometrize-go/internal/model"
	"forza-painter-geometrize-go/internal/output"
	"forza-painter-geometrize-go/internal/render"
)

type Options struct {
	ImagePath     string
	SettingsPath  string
	Profile       string
	OutputPath    string
	PreviewPath   string
	WorkspaceRoot string
	Seed          int64
}

const (
	maxNoImproveRetries = 100
	minImproveDelta     = -1e-7

	// Hill climb tuning. The mutation budget from settings is split into
	// up to maxHillClimbRounds rounds; each round mutates the current best
	// shape geometry slightly, evaluates the batch on GPU, and keeps any
	// improvement before starting the next round. This is the standard
	// geometrize-style hill climb adapted to GPU-friendly batches.
	maxHillClimbRounds  = 32
	idealHillClimbBatch = 64
	minHillClimbRounds  = 1
)

func Run(opts Options) error {
	if opts.ImagePath == "" {
		return fmt.Errorf("image path is required")
	}
	if opts.WorkspaceRoot == "" {
		opts.WorkspaceRoot = "."
	}

	settingsPath, err := config.ResolveSettingsPath(opts.WorkspaceRoot, opts.SettingsPath, opts.Profile)
	if err != nil {
		return err
	}
	cfg, err := config.ParseSettings(settingsPath)
	if err != nil {
		return err
	}

	prepared, err := imageutil.LoadAndPrepare(opts.ImagePath, cfg.MaxResolution)
	if err != nil {
		return err
	}

	maxBatch := cfg.RandomSamples
	if cfg.MutatedSamples > maxBatch {
		maxBatch = cfg.MutatedSamples
	}
	evaluator, err := gpu.NewEvaluator(prepared.Target, prepared.Current, prepared.OpaqueMask, prepared.Width, prepared.Height, maxBatch)
	if err != nil {
		return err
	}
	defer evaluator.Close()

	rng := rand.New(rand.NewSource(seedValue(opts.Seed)))
	currentError, opaquePixels := computeTotalError(prepared.Target, prepared.Current, prepared.OpaqueMask)
	denom := float64(maxInt(1, opaquePixels*4))

	shapes := []model.Shape{backgroundShape(prepared, normalizeScore(currentError, denom))}

	moveStep, radiusStep := mutationSteps(prepared.Width, prepared.Height)

	hillClimbRounds, mutationsPerRound := planHillClimb(cfg.MutatedSamples)

	sampler, err := refreshSampler(evaluator, prepared)
	if err != nil {
		return err
	}

	fmt.Printf("Loaded image: %s (%dx%d), transparency=%v\n", opts.ImagePath, prepared.Width, prepared.Height, prepared.HasTransparency)
	fmt.Printf("Settings: stopAt=%d randomSamples=%d mutatedSamples=%d saveAt=%d saveEvery(preview)=%d\n",
		cfg.StopAt, cfg.RandomSamples, cfg.MutatedSamples, len(cfg.SaveAt), cfg.SaveEvery)
	fmt.Printf("Compatibility mode: forceOpaqueShapes=%v\n", cfg.ForceOpaqueShapes)
	fmt.Printf("Hill climb: %d rounds x %d mutations (move +/- %.1fpx, radius +/- %.1fpx, theta +/- 30deg)\n",
		hillClimbRounds, mutationsPerRound, moveStep, radiusStep)
	fmt.Println("Scoring mode: delta error with GPU-computed optimal color (negative = better)")

	acceptedShapes := 0
	consecutiveNoImprove := 0

	for acceptedShapes < cfg.StopAt {
		step := acceptedShapes + 1
		stepStart := time.Now()
		fmt.Printf("[%d/%d] Generating random samples (%d)...\n", step, cfg.StopAt, cfg.RandomSamples)
		randomCands := randomCandidates(rng, prepared, cfg.RandomSamples, cfg.ForceOpaqueShapes, sampler)
		fmt.Printf("[%d/%d] Evaluating random sample batch on OpenCL (%d)...\n", step, cfg.StopAt, len(randomCands))
		best, bestScore, err := evaluateBest(evaluator, randomCands)
		if err != nil {
			return err
		}
		fmt.Printf("[%d/%d] Random best delta: %.6f\n", step, cfg.StopAt, bestScore)

		if hillClimbRounds > 0 && mutationsPerRound > 0 && bestScore < 0 {
			improved := 0
			for round := 0; round < hillClimbRounds; round++ {
				mutations := mutatedCandidates(rng, prepared, best, mutationsPerRound, cfg.ForceOpaqueShapes, moveStep, radiusStep)
				roundBest, roundScore, mutErr := evaluateBest(evaluator, mutations)
				if mutErr != nil {
					return mutErr
				}
				if roundScore < bestScore {
					bestScore = roundScore
					best = roundBest
					improved++
				}
			}
			fmt.Printf("[%d/%d] Hill climb best delta after %d rounds: %.6f (%d improvement(s))\n",
				step, cfg.StopAt, hillClimbRounds, bestScore, improved)
		}

		if bestScore >= minImproveDelta {
			consecutiveNoImprove++
			fmt.Printf("[%d/%d] No improvement (delta %.6f). Retry %d/%d\n", step, cfg.StopAt, bestScore, consecutiveNoImprove, maxNoImproveRetries)
			if consecutiveNoImprove >= maxNoImproveRetries {
				fmt.Printf("Stopped early: reached max retries without improvement (%d)\n", maxNoImproveRetries)
				break
			}
			continue
		}

		consecutiveNoImprove = 0

		// Quantize geometry + colour to the integer grid that will end up in
		// the JSON; apply that exact shape so the GPU canvas matches what the
		// game will render from the JSON later.
		final := quantizeCandidate(best, prepared.Width, prepared.Height, cfg.ForceOpaqueShapes)

		if err := evaluator.Apply(final); err != nil {
			return err
		}
		currentError += float64(bestScore)
		if currentError < 0 {
			currentError = 0
		}
		shapes = append(shapes, toShape(final, normalizeScore(currentError, denom)))
		acceptedShapes++
		fmt.Printf("[%d/%d] Added rotated ellipse #%d (delta %.6f)\n", acceptedShapes, cfg.StopAt, len(shapes)-1, bestScore)

		// Refresh the error histogram for biased sampling on the next step.
		newSampler, sErr := refreshSampler(evaluator, prepared)
		if sErr != nil {
			return sErr
		}
		sampler = newSampler

		if shouldSave(acceptedShapes, cfg) {
			if err := saveShapes(opts, shapes, acceptedShapes); err != nil {
				return err
			}
			fmt.Printf("[%d/%d] Saved geometry checkpoint for shape count %d\n", acceptedShapes, cfg.StopAt, acceptedShapes)
		}

		if shouldSavePreview(acceptedShapes, cfg) {
			if err := savePreviewSnapshot(evaluator, opts, prepared.Width, prepared.Height, acceptedShapes); err != nil {
				return err
			}
			if opts.PreviewPath != "" {
				fmt.Printf("[%d/%d] Saved preview snapshot\n", acceptedShapes, cfg.StopAt)
			}
		}

		fmt.Printf("[%d/%d] Step completed in %s\n", acceptedShapes, cfg.StopAt, time.Since(stepStart).Round(time.Millisecond))
	}

	if acceptedShapes < cfg.StopAt {
		fmt.Printf("Finished early with %d/%d shapes due to no-improvement stopping rule\n", acceptedShapes, cfg.StopAt)
	}

	if err := output.SaveGeometry(output.BuildFinalOutputPath(resolveOutputBase(opts)), shapes); err != nil {
		return err
	}

	if opts.PreviewPath != "" {
		current := make([]float32, prepared.Width*prepared.Height*4)
		if err := evaluator.ReadCurrent(current); err != nil {
			return err
		}
		if err := render.SavePNG(opts.PreviewPath, current, prepared.Width, prepared.Height); err != nil {
			return err
		}
	}

	return nil
}

func seedValue(seed int64) int64 {
	if seed != 0 {
		return seed
	}
	return time.Now().UnixNano()
}

func backgroundShape(p *imageutil.PreparedImage, score float64) model.Shape {
	return model.Shape{
		Type:  1,
		Data:  []int{0, 0, p.Width, p.Height},
		Color: []int{int(p.BackgroundRGBA[0]), int(p.BackgroundRGBA[1]), int(p.BackgroundRGBA[2]), int(p.BackgroundRGBA[3])},
		Score: score,
	}
}

// planHillClimb splits the configured mutation budget into a number of
// rounds and a per-round batch size. Each round will be a single GPU
// dispatch and the best shape from that dispatch becomes the seed of the
// next round (real hill climbing). We aim for ~64 candidates per round to
// keep the GPU occupied while still giving the climb enough steps to walk
// uphill instead of just sampling around the random seed.
func planHillClimb(budget int) (rounds, perRound int) {
	if budget <= 0 {
		return 0, 0
	}
	rounds = budget / idealHillClimbBatch
	if rounds < minHillClimbRounds {
		rounds = minHillClimbRounds
	}
	if rounds > maxHillClimbRounds {
		rounds = maxHillClimbRounds
	}
	perRound = budget / rounds
	if perRound < 1 {
		perRound = 1
	}
	return rounds, perRound
}

func mutationSteps(width, height int) (move, radius float32) {
	diag := math.Sqrt(float64(width*width) + float64(height*height))
	move = float32(math.Max(2.0, diag*0.012))
	radius = float32(math.Max(2.0, diag*0.010))
	return move, radius
}

// errorSampler converts the GPU-produced error histogram into a CDF that
// can be sampled in O(log n) per draw. It is rebuilt every accepted shape.
type errorSampler struct {
	gridW, gridH int
	imgW, imgH   int
	cdf          []float64
	total        float64
}

func newErrorSampler(grid []float32, gridW, gridH, imgW, imgH int) *errorSampler {
	cdf := make([]float64, len(grid))
	var total float64
	for i, v := range grid {
		if v < 0 {
			v = 0
		}
		total += float64(v)
		cdf[i] = total
	}
	return &errorSampler{
		gridW: gridW,
		gridH: gridH,
		imgW:  imgW,
		imgH:  imgH,
		cdf:   cdf,
		total: total,
	}
}

func (s *errorSampler) sample(rng *rand.Rand) (float32, float32) {
	if s == nil || s.total <= 0 || s.gridW <= 0 || s.gridH <= 0 {
		return rng.Float32() * float32(s.imgW), rng.Float32() * float32(s.imgH)
	}
	u := rng.Float64() * s.total
	lo, hi := 0, len(s.cdf)-1
	for lo < hi {
		mid := (lo + hi) / 2
		if s.cdf[mid] < u {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	cell := lo
	gx := cell % s.gridW
	gy := cell / s.gridW
	x0 := int(int64(gx) * int64(s.imgW) / int64(s.gridW))
	x1 := int(int64(gx+1) * int64(s.imgW) / int64(s.gridW))
	y0 := int(int64(gy) * int64(s.imgH) / int64(s.gridH))
	y1 := int(int64(gy+1) * int64(s.imgH) / int64(s.gridH))
	if x1 <= x0 {
		x1 = x0 + 1
	}
	if y1 <= y0 {
		y1 = y0 + 1
	}
	if x1 > s.imgW {
		x1 = s.imgW
	}
	if y1 > s.imgH {
		y1 = s.imgH
	}
	x := float32(x0) + rng.Float32()*float32(x1-x0)
	y := float32(y0) + rng.Float32()*float32(y1-y0)
	return x, y
}

func refreshSampler(evaluator *gpu.Evaluator, prepared *imageutil.PreparedImage) (*errorSampler, error) {
	grid, gw, gh, err := evaluator.ErrorGrid()
	if err != nil {
		return nil, err
	}
	return newErrorSampler(grid, gw, gh, prepared.Width, prepared.Height), nil
}

// randomCandidates seeds candidates whose CENTER is biased towards the
// regions of the image that still have the most error. Geometry (radius,
// angle) is randomized; color is left zero because the GPU evaluator
// computes the optimal color analytically and writes it back in the
// EvalResult.
func randomCandidates(rng *rand.Rand, prepared *imageutil.PreparedImage, count int, forceOpaque bool, sampler *errorSampler) []model.Candidate {
	out := make([]model.Candidate, 0, count)
	w := float32(prepared.Width)
	h := float32(prepared.Height)
	diag := float32(math.Sqrt(float64(prepared.Width*prepared.Width) + float64(prepared.Height*prepared.Height)))
	maxRadius := diag * 0.25
	if maxRadius < 4 {
		maxRadius = 4
	}
	minRadius := float32(2)
	maxAttempts := count * 4
	attempts := 0

	for len(out) < count && attempts < maxAttempts {
		attempts++
		x, y := sampler.sample(rng)
		if x < 0 {
			x = 0
		}
		if y < 0 {
			y = 0
		}
		if x > w-1 {
			x = w - 1
		}
		if y > h-1 {
			y = h - 1
		}
		alpha := float32(1.0)
		if !forceOpaque {
			alpha = randRange(rng, 0.3, 1.0)
		}
		out = append(out, model.Candidate{
			X:     x,
			Y:     y,
			RX:    randRange(rng, minRadius, maxRadius),
			RY:    randRange(rng, minRadius, maxRadius),
			Theta: rng.Float32() * 360,
			A:     alpha,
		})
	}
	if len(out) == 0 {
		out = append(out, model.Candidate{
			X:     w * 0.5,
			Y:     h * 0.5,
			RX:    maxRadius * 0.25,
			RY:    maxRadius * 0.25,
			Theta: 0,
			A:     1.0,
		})
	}
	return out
}

// mutatedCandidates only perturbs geometry. Colors are recomputed by the
// GPU on each evaluation, so seeding them on the CPU side would be wasted
// work (and would constrain the search).
func mutatedCandidates(rng *rand.Rand, prepared *imageutil.PreparedImage, base model.Candidate, count int, forceOpaque bool, moveStep, radiusStep float32) []model.Candidate {
	out := make([]model.Candidate, 0, count)
	w := float32(prepared.Width)
	h := float32(prepared.Height)
	for i := 0; i < count; i++ {
		cand := base
		cand.X += randRange(rng, -moveStep, moveStep)
		cand.Y += randRange(rng, -moveStep, moveStep)
		if cand.X < 0 {
			cand.X = 0
		}
		if cand.Y < 0 {
			cand.Y = 0
		}
		if cand.X > w-1 {
			cand.X = w - 1
		}
		if cand.Y > h-1 {
			cand.Y = h - 1
		}
		cand.RX = float32(math.Max(1, float64(cand.RX+randRange(rng, -radiusStep, radiusStep))))
		cand.RY = float32(math.Max(1, float64(cand.RY+randRange(rng, -radiusStep, radiusStep))))
		cand.Theta += randRange(rng, -30, 30)
		if cand.Theta < 0 {
			cand.Theta += 360
		}
		if cand.Theta >= 360 {
			cand.Theta -= 360
		}
		if forceOpaque {
			cand.A = 1.0
		}
		out = append(out, cand)
	}
	if len(out) == 0 {
		out = append(out, base)
	}
	return out
}

// evaluateBest dispatches the batch on the GPU, picks the lowest-score
// candidate and merges the GPU-computed optimal color into it so that
// subsequent hill-climb rounds work from the correct base color.
func evaluateBest(e *gpu.Evaluator, cands []model.Candidate) (model.Candidate, float32, error) {
	results, err := e.Evaluate(cands)
	if err != nil {
		return model.Candidate{}, 0, err
	}
	if len(results) == 0 {
		return model.Candidate{}, 0, fmt.Errorf("no candidate scores returned")
	}
	bestIdx := 0
	bestScore := results[0].Score
	for i := 1; i < len(results); i++ {
		if results[i].Score < bestScore {
			bestScore = results[i].Score
			bestIdx = i
		}
	}
	best := cands[bestIdx]
	best.R = results[bestIdx].R
	best.G = results[bestIdx].G
	best.B = results[bestIdx].B
	return best, bestScore, nil
}

func toShape(c model.Candidate, score float64) model.Shape {
	angle := int(math.Round(float64(c.Theta))) % 360
	if angle < 0 {
		angle += 360
	}
	if angle == 0 && c.Theta > 359.5 {
		angle = 360
	}
	return model.Shape{
		Type: 16,
		Data: []int{
			int(math.Round(float64(c.X))),
			int(math.Round(float64(c.Y))),
			maxInt(1, int(math.Round(float64(c.RX)))),
			maxInt(1, int(math.Round(float64(c.RY)))),
			angle,
		},
		Color: []int{int(f32ToByte(c.R)), int(f32ToByte(c.G)), int(f32ToByte(c.B)), int(f32ToByte(c.A))},
		Score: score,
	}
}

func f32ToByte(v float32) uint8 {
	if v < 0 {
		v = 0
	}
	if v > 1 {
		v = 1
	}
	return uint8(math.Round(float64(v * 255)))
}

func shouldSave(step int, cfg model.Settings) bool {
	_, ok := cfg.SaveAt[step]
	return ok
}

func shouldSavePreview(step int, cfg model.Settings) bool {
	if cfg.SaveEvery < 1 {
		return false
	}
	return step%cfg.SaveEvery == 0
}

func saveShapes(opts Options, shapes []model.Shape, step int) error {
	base := resolveOutputBase(opts)
	return output.SaveGeometry(output.BuildOutputPath(base, step), shapes)
}

func resolveOutputBase(opts Options) string {
	if opts.OutputPath != "" {
		return opts.OutputPath
	}
	ext := filepath.Ext(opts.ImagePath)
	if ext == "" {
		return opts.ImagePath
	}
	return opts.ImagePath
}

func randRange(rng *rand.Rand, minV, maxV float32) float32 {
	return minV + (maxV-minV)*rng.Float32()
}

func savePreviewSnapshot(evaluator *gpu.Evaluator, opts Options, width, height, step int) error {
	if opts.PreviewPath == "" {
		return nil
	}
	ext := filepath.Ext(opts.PreviewPath)
	base := opts.PreviewPath
	if ext != "" {
		base = opts.PreviewPath[:len(opts.PreviewPath)-len(ext)]
	}
	outPath := fmt.Sprintf("%s.%d.png", base, step)
	current := make([]float32, width*height*4)
	if err := evaluator.ReadCurrent(current); err != nil {
		return err
	}
	return render.SavePNG(outPath, current, width, height)
}

func computeTotalError(target, current []float32, opaqueMask []uint8) (float64, int) {
	if len(target) != len(current) {
		return 0, 0
	}
	total := 0.0
	opaquePixels := 0
	for p := 0; p < len(opaqueMask); p++ {
		if opaqueMask[p] == 0 {
			continue
		}
		opaquePixels++
		idx := p * 4
		dr := float64(target[idx+0] - current[idx+0])
		dg := float64(target[idx+1] - current[idx+1])
		db := float64(target[idx+2] - current[idx+2])
		da := float64(target[idx+3] - current[idx+3])
		total += dr*dr + dg*dg + db*db + da*da
	}
	return total, opaquePixels
}

func normalizeScore(totalError, denom float64) float64 {
	if denom <= 0 {
		return 0
	}
	value := totalError / denom
	if value < 0 {
		value = 0
	}
	return math.Round(value*1_000_000) / 1_000_000
}

// quantizeCandidate is now only invoked at acceptance time. The GPU search
// runs on full-precision floats; the final shape that is committed both to
// the canvas and to the JSON gets snapped to the integer grid the game
// expects (pixel positions, integer angle, 8-bit colour).
func quantizeCandidate(c model.Candidate, width, height int, forceOpaque bool) model.Candidate {
	c.X = float32(clampInt(int(math.Round(float64(c.X))), 0, maxInt(0, width-1)))
	c.Y = float32(clampInt(int(math.Round(float64(c.Y))), 0, maxInt(0, height-1)))
	c.RX = float32(maxInt(1, int(math.Round(float64(c.RX)))))
	c.RY = float32(maxInt(1, int(math.Round(float64(c.RY)))))

	angle := int(math.Round(float64(c.Theta))) % 360
	if angle < 0 {
		angle += 360
	}
	if angle == 0 && c.Theta > 359.5 {
		angle = 360
	}
	c.Theta = float32(angle)

	if forceOpaque {
		c.A = 1.0
	}
	c.R = float32(f32ToByte(c.R)) / 255.0
	c.G = float32(f32ToByte(c.G)) / 255.0
	c.B = float32(f32ToByte(c.B)) / 255.0
	c.A = float32(f32ToByte(c.A)) / 255.0
	return c
}

func clampInt(v, minV, maxV int) int {
	if v < minV {
		return minV
	}
	if v > maxV {
		return maxV
	}
	return v
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
