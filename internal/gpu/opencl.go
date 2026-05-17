package gpu

import (
	"fmt"
	"math"
	"unsafe"

	"forza-painter-geometrize-go/internal/model"

	"github.com/jgillich/go-opencl/cl"
)

// ErrorGridSize is the side length of the downsampled error histogram used
// for biasing random candidate placement towards high-error regions.
const ErrorGridSize = 64

// EvalResult holds the score and the optimal RGB color for a single
// evaluated candidate. RGB is computed analytically by the GPU, the engine
// stores it back into the candidate before applying the chosen shape.
type EvalResult struct {
	Score float32
	R     float32
	G     float32
	B     float32
}

type Evaluator struct {
	context        *cl.Context
	queue          *cl.CommandQueue
	program        *cl.Program
	evalKernel     *cl.Kernel
	applyKernel    *cl.Kernel
	gridKernel     *cl.Kernel
	targetBuffer   *cl.MemObject
	currentBuffer  *cl.MemObject
	maskBuffer     *cl.MemObject
	candBuffer     *cl.MemObject
	resultBuffer   *cl.MemObject
	errorGridBuf   *cl.MemObject
	width          int
	height         int
	pixelCount     int
	maxCandidates  int
	hostPacked     []float32
	hostResults    []float32
	gridW          int
	gridH          int
	hostErrorGrid  []float32
}

func NewEvaluator(target, current []float32, mask []uint8, width, height int, maxCandidates int) (*Evaluator, error) {
	if len(target) != len(current) {
		return nil, fmt.Errorf("target/current length mismatch")
	}
	if len(mask) != width*height {
		return nil, fmt.Errorf("mask length mismatch")
	}
	if maxCandidates < 1 {
		return nil, fmt.Errorf("maxCandidates must be > 0")
	}

	platforms, err := cl.GetPlatforms()
	if err != nil {
		return nil, err
	}
	if len(platforms) == 0 {
		return nil, fmt.Errorf("no OpenCL platforms found")
	}

	var device *cl.Device
	for _, p := range platforms {
		devices, dErr := p.GetDevices(cl.DeviceTypeGPU)
		if dErr == nil && len(devices) > 0 {
			device = devices[0]
			break
		}
	}
	if device == nil {
		for _, p := range platforms {
			devices, dErr := p.GetDevices(cl.DeviceTypeAll)
			if dErr == nil && len(devices) > 0 {
				device = devices[0]
				break
			}
		}
	}
	if device == nil {
		return nil, fmt.Errorf("no OpenCL device found")
	}

	ctx, err := cl.CreateContext([]*cl.Device{device})
	if err != nil {
		return nil, err
	}

	queue, err := ctx.CreateCommandQueue(device, 0)
	if err != nil {
		ctx.Release()
		return nil, err
	}

	program, err := ctx.CreateProgramWithSource([]string{evaluateKernelSource})
	if err != nil {
		queue.Release()
		ctx.Release()
		return nil, err
	}
	if err := program.BuildProgram([]*cl.Device{device}, "-cl-fast-relaxed-math -cl-mad-enable"); err != nil {
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, fmt.Errorf("failed building OpenCL program: %w", err)
	}

	evalKernel, err := program.CreateKernel("evaluate_candidates_v3")
	if err != nil {
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	applyKernel, err := program.CreateKernel("apply_candidate_v2")
	if err != nil {
		evalKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	gridKernel, err := program.CreateKernel("compute_error_grid")
	if err != nil {
		applyKernel.Release()
		evalKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}

	gridW := ErrorGridSize
	gridH := ErrorGridSize
	if width < gridW {
		gridW = width
	}
	if height < gridH {
		gridH = height
	}
	if gridW < 1 {
		gridW = 1
	}
	if gridH < 1 {
		gridH = 1
	}

	targetBuffer, err := ctx.CreateEmptyBuffer(cl.MemReadOnly, len(target)*4)
	if err != nil {
		gridKernel.Release()
		applyKernel.Release()
		evalKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	currentBuffer, err := ctx.CreateEmptyBuffer(cl.MemReadWrite, len(current)*4)
	if err != nil {
		targetBuffer.Release()
		gridKernel.Release()
		applyKernel.Release()
		evalKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	maskBuffer, err := ctx.CreateEmptyBuffer(cl.MemReadOnly, len(mask))
	if err != nil {
		currentBuffer.Release()
		targetBuffer.Release()
		gridKernel.Release()
		applyKernel.Release()
		evalKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	candBuffer, err := ctx.CreateEmptyBuffer(cl.MemReadOnly, maxCandidates*6*4)
	if err != nil {
		maskBuffer.Release()
		currentBuffer.Release()
		targetBuffer.Release()
		gridKernel.Release()
		applyKernel.Release()
		evalKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	resultBuffer, err := ctx.CreateEmptyBuffer(cl.MemWriteOnly, maxCandidates*4*4)
	if err != nil {
		candBuffer.Release()
		maskBuffer.Release()
		currentBuffer.Release()
		targetBuffer.Release()
		gridKernel.Release()
		applyKernel.Release()
		evalKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}
	errorGridBuf, err := ctx.CreateEmptyBuffer(cl.MemReadWrite, gridW*gridH*4)
	if err != nil {
		resultBuffer.Release()
		candBuffer.Release()
		maskBuffer.Release()
		currentBuffer.Release()
		targetBuffer.Release()
		gridKernel.Release()
		applyKernel.Release()
		evalKernel.Release()
		program.Release()
		queue.Release()
		ctx.Release()
		return nil, err
	}

	if _, err := queue.EnqueueWriteBufferFloat32(targetBuffer, true, 0, target, nil); err != nil {
		return nil, err
	}
	if _, err := queue.EnqueueWriteBufferFloat32(currentBuffer, true, 0, current, nil); err != nil {
		return nil, err
	}
	if _, err := queue.EnqueueWriteBuffer(maskBuffer, true, 0, len(mask), unsafe.Pointer(&mask[0]), nil); err != nil {
		return nil, err
	}

	return &Evaluator{
		context:       ctx,
		queue:         queue,
		program:       program,
		evalKernel:    evalKernel,
		applyKernel:   applyKernel,
		gridKernel:    gridKernel,
		targetBuffer:  targetBuffer,
		currentBuffer: currentBuffer,
		maskBuffer:    maskBuffer,
		candBuffer:    candBuffer,
		resultBuffer:  resultBuffer,
		errorGridBuf:  errorGridBuf,
		width:         width,
		height:        height,
		pixelCount:    width * height,
		maxCandidates: maxCandidates,
		hostPacked:    make([]float32, maxCandidates*6),
		hostResults:   make([]float32, maxCandidates*4),
		gridW:         gridW,
		gridH:         gridH,
		hostErrorGrid: make([]float32, gridW*gridH),
	}, nil
}

func (e *Evaluator) Close() {
	if e.errorGridBuf != nil {
		e.errorGridBuf.Release()
	}
	if e.resultBuffer != nil {
		e.resultBuffer.Release()
	}
	if e.candBuffer != nil {
		e.candBuffer.Release()
	}
	if e.maskBuffer != nil {
		e.maskBuffer.Release()
	}
	if e.currentBuffer != nil {
		e.currentBuffer.Release()
	}
	if e.targetBuffer != nil {
		e.targetBuffer.Release()
	}
	if e.gridKernel != nil {
		e.gridKernel.Release()
	}
	if e.applyKernel != nil {
		e.applyKernel.Release()
	}
	if e.evalKernel != nil {
		e.evalKernel.Release()
	}
	if e.program != nil {
		e.program.Release()
	}
	if e.queue != nil {
		e.queue.Release()
	}
	if e.context != nil {
		e.context.Release()
	}
}

// Evaluate dispatches the candidate batch to the GPU and returns one
// EvalResult per candidate (score + analytically computed optimal color).
func (e *Evaluator) Evaluate(candidates []model.Candidate) ([]EvalResult, error) {
	count := len(candidates)
	if count == 0 {
		return nil, nil
	}
	if count > e.maxCandidates {
		return nil, fmt.Errorf("candidate count %d exceeds max %d", count, e.maxCandidates)
	}

	packed := e.hostPacked[:count*6]
	for i, c := range candidates {
		base := i * 6
		packed[base+0] = c.X
		packed[base+1] = c.Y
		packed[base+2] = c.RX
		packed[base+3] = c.RY
		packed[base+4] = c.Theta
		packed[base+5] = c.A
	}

	if _, err := e.queue.EnqueueWriteBufferFloat32(e.candBuffer, true, 0, packed, nil); err != nil {
		return nil, err
	}

	if err := e.evalKernel.SetArgs(
		e.targetBuffer,
		e.currentBuffer,
		e.maskBuffer,
		e.candBuffer,
		e.resultBuffer,
		int32(e.width),
		int32(e.height),
	); err != nil {
		return nil, err
	}

	globalSize := []int{count}
	if _, err := e.queue.EnqueueNDRangeKernel(e.evalKernel, nil, globalSize, nil, nil); err != nil {
		return nil, err
	}

	flat := e.hostResults[:count*4]
	if _, err := e.queue.EnqueueReadBufferFloat32(e.resultBuffer, true, 0, flat, nil); err != nil {
		return nil, err
	}
	out := make([]EvalResult, count)
	for i := 0; i < count; i++ {
		out[i] = EvalResult{
			Score: flat[i*4+0],
			R:     flat[i*4+1],
			G:     flat[i*4+2],
			B:     flat[i*4+3],
		}
	}
	return out, nil
}

// Apply blends the (already chosen) candidate into the current canvas.
// The kernel only runs on the candidate's bounding box.
func (e *Evaluator) Apply(candidate model.Candidate) error {
	rx := candidate.RX
	ry := candidate.RY
	if rx < 1 {
		rx = 1
	}
	if ry < 1 {
		ry = 1
	}
	theta := float64(candidate.Theta) * (math.Pi / 180.0)
	cosT := math.Cos(theta)
	sinT := math.Sin(theta)
	rx2 := float64(rx) * float64(rx)
	ry2 := float64(ry) * float64(ry)
	cos2 := cosT * cosT
	sin2 := sinT * sinT
	ex := math.Sqrt(rx2*cos2 + ry2*sin2)
	ey := math.Sqrt(rx2*sin2 + ry2*cos2)

	xMin := int(math.Floor(float64(candidate.X) - ex - 1.0))
	xMax := int(math.Ceil(float64(candidate.X) + ex + 1.0))
	yMin := int(math.Floor(float64(candidate.Y) - ey - 1.0))
	yMax := int(math.Ceil(float64(candidate.Y) + ey + 1.0))

	if xMin < 0 {
		xMin = 0
	}
	if yMin < 0 {
		yMin = 0
	}
	if xMax > e.width-1 {
		xMax = e.width - 1
	}
	if yMax > e.height-1 {
		yMax = e.height - 1
	}
	if xMax < xMin || yMax < yMin {
		return nil
	}

	bw := xMax - xMin + 1
	bh := yMax - yMin + 1

	if err := e.applyKernel.SetArgs(
		e.currentBuffer,
		e.maskBuffer,
		int32(e.width),
		int32(e.height),
		int32(xMin),
		int32(yMin),
		int32(xMax),
		int32(yMax),
		candidate.X,
		candidate.Y,
		candidate.RX,
		candidate.RY,
		candidate.Theta,
		candidate.R,
		candidate.G,
		candidate.B,
		candidate.A,
	); err != nil {
		return err
	}

	globalSize := []int{bw, bh}
	if _, err := e.queue.EnqueueNDRangeKernel(e.applyKernel, nil, globalSize, nil, nil); err != nil {
		return err
	}
	return nil
}

func (e *Evaluator) ReadCurrent(dst []float32) error {
	if len(dst) != e.pixelCount*4 {
		return fmt.Errorf("destination length mismatch")
	}
	_, err := e.queue.EnqueueReadBufferFloat32(e.currentBuffer, true, 0, dst, nil)
	return err
}

// ErrorGrid recomputes the downsampled per-cell squared error map and
// returns it together with its dimensions. Cells are summed over the
// pixels they cover; transparent (masked) pixels are ignored.
func (e *Evaluator) ErrorGrid() ([]float32, int, int, error) {
	if err := e.gridKernel.SetArgs(
		e.targetBuffer,
		e.currentBuffer,
		e.maskBuffer,
		e.errorGridBuf,
		int32(e.width),
		int32(e.height),
		int32(e.gridW),
		int32(e.gridH),
	); err != nil {
		return nil, 0, 0, err
	}
	globalSize := []int{e.gridW, e.gridH}
	if _, err := e.queue.EnqueueNDRangeKernel(e.gridKernel, nil, globalSize, nil, nil); err != nil {
		return nil, 0, 0, err
	}
	if _, err := e.queue.EnqueueReadBufferFloat32(e.errorGridBuf, true, 0, e.hostErrorGrid, nil); err != nil {
		return nil, 0, 0, err
	}
	out := make([]float32, len(e.hostErrorGrid))
	copy(out, e.hostErrorGrid)
	return out, e.gridW, e.gridH, nil
}

// GridDims returns the dimensions of the error histogram grid.
func (e *Evaluator) GridDims() (int, int) {
	return e.gridW, e.gridH
}

// ImageDims returns the working image dimensions.
func (e *Evaluator) ImageDims() (int, int) {
	return e.width, e.height
}
