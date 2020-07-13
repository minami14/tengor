package nn

import (
	"fmt"
	"reflect"
	"time"
)

// Model is a neural network model.
type Model interface {
	Layers() []Layer
	Fit(x, y []*Tensor, epochs, batchSize int)
	Predict([]*Tensor) []*Tensor
	Build(Loss) error
}

// Sequential is a model that stack of layers.
type Sequential struct {
	inputShape       Shape
	outputShape      Shape
	layers           []Layer
	loss             Loss
	optimizerFactory OptimizerFactory
}

// NewSequential creates an instance of sequential model.
func NewSequential(inputShape Shape) *Sequential {
	return &Sequential{
		inputShape:  inputShape,
		outputShape: inputShape,
		layers:      []Layer{&inputLayer{}},
	}
}

// Layers returns layers that model has.
func (s *Sequential) Layers() []Layer {
	return s.layers
}

// Fit fits the model to the given dataset.
func (s *Sequential) Fit(x, t []*Tensor, epochs, batchSize int) {
	totalStart := time.Now()
	for epoch := 0; epoch < epochs; epoch++ {
		fmt.Printf("epoch %v/%v\n", epoch+1, epochs)
		steps := len(x) / batchSize
		start := time.Now()
		for step := 0; step < steps; step++ {
			startIndex := step * batchSize
			endIndex := (step + 1) * batchSize
			y := s.Predict(x[startIndex:endIndex])
			loss := s.Loss(y, t[startIndex:endIndex])
			acc := s.Accuracy(y, t[startIndex:endIndex])
			fmt.Printf("\r\033[K%v/%v\t%v%%\t%.1fs\tloss: %.4f\tacc: %.4f", step*batchSize, steps*batchSize, 100*step/steps, time.Now().Sub(start).Seconds(), loss, acc)
			s.update(x[startIndex:endIndex], t[startIndex:endIndex])
		}
		y := s.Predict(x)
		loss := s.Loss(y, t)
		acc := s.Accuracy(y, t)
		fmt.Printf("\r\033[K%v/%v\t100%%\t%.1fs\tloss: %.4f\tacc: %.4f\n", steps*batchSize, steps*batchSize, time.Now().Sub(start).Seconds(), loss, acc)
	}
	fmt.Printf("%.1fs\n", time.Now().Sub(totalStart).Seconds())
}

func (s *Sequential) update(x, t []*Tensor) {
	for _, layer := range s.layers {
		x = layer.Forward(x)
	}

	s.loss.Forward(x, t)
	dout := s.loss.Backward()
	for i := len(s.layers) - 1; i >= 0; i-- {
		dout = s.layers[i].Backward(dout)
		s.layers[i].Update()
	}
}

// Predict predicts output for the given data.
func (s *Sequential) Predict(inputs []*Tensor) []*Tensor {
	x := inputs
	for _, layer := range s.layers {
		x = layer.Call(x)
	}
	return x
}

// Loss is loss of predicted value.
func (s *Sequential) Loss(y, t []*Tensor) float64 {
	return s.loss.Call(y, t)
}

// Accuracy is accuracy of predicted value.
func (s *Sequential) Accuracy(y, t []*Tensor) float64 {
	sum := 0.0
	for i := 0; i < len(t); i++ {
		if y[i].MaxIndex() == t[i].MaxIndex() {
			sum++
		}
	}
	return sum / float64(len(t))
}

// Build builds a model by connecting the given layers.
func (s *Sequential) Build(loss Loss, factory OptimizerFactory) error {
	if err := s.layers[0].Init(s.inputShape, factory); err != nil {
		return err
	}

	shape := s.layers[0].OutputShape()
	for i, layer := range s.layers[1:] {
		if err := layer.Init(shape, factory); err != nil {
			return fmt.Errorf("build error layer %v %v %v", i+1, reflect.TypeOf(layer), err)
		}

		shape = layer.OutputShape()
	}

	s.loss = loss
	s.optimizerFactory = factory

	return nil
}

// AddLayer adds layer to model.
func (s *Sequential) AddLayer(layer Layer) {
	s.layers = append(s.layers, layer)
}

// Summary is summary of model.
func (s *Sequential) Summary() string {
	res := "Layer Type\tOutput Shape\tParams\n=======================================\n"
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
