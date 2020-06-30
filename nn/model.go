package nn

import (
	"fmt"
	"reflect"
	"time"
)

type Model interface {
	Layers() []Layer
	Fit(x, y []*Tensor)
	Predict([]*Tensor) []*Tensor
	Build(Loss) error
}

type Sequential struct {
	inputShape   Shape
	outputShape  Shape
	layers       []Layer
	loss         Loss
	LearningRate float64
}

func NewSequential(inputShape Shape) *Sequential {
	return &Sequential{
		inputShape:   inputShape,
		outputShape:  inputShape,
		layers:       []Layer{&Input{}},
		LearningRate: 0.1,
	}
}

func (s *Sequential) Layers() []Layer {
	return s.layers
}

func (s *Sequential) Fit(x, y []*Tensor, epochs, batchSize int) {
	totalStart := time.Now()
	for epoch := 0; epoch < epochs; epoch++ {
		fmt.Printf("epoch %v/%v\n", epoch+1, epochs)
		steps := len(x) / batchSize
		start := time.Now()
		for step := 0; step < steps; step++ {
			fmt.Printf("\r\033[K%v/%v\t%v%%\t%.1fs", step*batchSize, steps*batchSize, 100*float64(step)/float64(steps), time.Now().Sub(start).Seconds())
			startIndex := step * batchSize
			endIndex := (step + 1) * batchSize
			s.Update(x[startIndex:endIndex], y[startIndex:endIndex])
		}
		loss := s.Loss(x, y)
		fmt.Printf("\r\033[K%v/%v\t100%%\t%.1fs\tloss: %v\n", steps, steps, time.Now().Sub(start).Seconds(), loss)
	}
	fmt.Printf("%.1fs\n", time.Now().Sub(totalStart).Seconds())
}

func (s *Sequential) Update(x, y []*Tensor) {
	const h = 1e-4
	grads := make([]*Tensor, 0)
	params := make([]*Tensor, 0)
	for _, layer := range s.layers {
		for _, param := range layer.Params() {
			params = append(params, param)
			grad := NewTensor(param.shape)
			for i := range param.rawData {
				tmp := param.rawData[i]
				param.rawData[i] += h
				h1 := s.Loss(x, y)
				param.rawData[i] = tmp - h
				h2 := s.Loss(x, y)
				param.rawData[i] = tmp
				grad.rawData[i] = (h1 - h2) / (2 * h)
			}
			grads = append(grads, grad)
		}
	}

	for i, grad := range grads {
		for j := range params[i].rawData {
			params[i].rawData[j] -= s.LearningRate * grad.rawData[j]
		}
	}
}

func (s *Sequential) Predict(inputs []*Tensor) []*Tensor {
	y := make([]*Tensor, len(inputs))
	for i, input := range inputs {
		x := input
		for _, layer := range s.layers {
			x = layer.Call(x)
		}
		y[i] = x
	}
	return y
}

func (s *Sequential) Loss(x, t []*Tensor) float64 {
	y := s.Predict(x)
	sum := 0.0
	for i, v := range y {
		sum += s.loss.Call(v, t[i])
	}

	loss := sum / float64(len(x))
	return loss
}

func (s *Sequential) Accuracy(x, y []*Tensor) float64 {
	return 0
}

func (s *Sequential) Build(loss Loss) error {
	if err := s.layers[0].Init(s.inputShape); err != nil {
		return err
	}

	shape := s.layers[0].OutputShape()
	for i, layer := range s.layers[1:] {
		if err := layer.Init(shape); err != nil {
			return fmt.Errorf("build error layer %v %v %v", i+1, reflect.TypeOf(layer), err)
		}

		shape = layer.OutputShape()
	}

	s.loss = loss

	return nil
}

func (s *Sequential) AddLayer(layer Layer) {
	s.layers = append(s.layers, layer)
}

func (s *Sequential) Summary() string {
	res := "Layer Type\tOutput Shape\tParams\n======================================================\n"
	sum := 0
	for _, layer := range s.layers {
		params := layer.Params()
		param := 0
		for _, p := range params {
			param += p.Shape().Elements()
		}

		res += fmt.Sprintf("%v\t\t%v\t\t%v\n", reflect.TypeOf(layer).String()[4:], layer.OutputShape(), param)
		sum += param
	}
	res += fmt.Sprintf("\nTotal params:\t%v", sum)
	return res
}
