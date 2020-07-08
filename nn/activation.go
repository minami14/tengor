package nn

import (
	"fmt"
	"math"
	"sync"
)

type relu struct {
	inputShape  Shape
	outputShape Shape
	mask        [][]bool
}

// ReLu is an activation function layer.
func ReLU() Layer {
	return &relu{}
}

func (r *relu) Init(inputShape Shape, _ OptimizerFactory) error {
	r.inputShape = inputShape
	r.outputShape = inputShape
	return nil
}

func (r *relu) Call(inputs []*Tensor) []*Tensor {
	outputs := make([]*Tensor, len(inputs))
	wg := new(sync.WaitGroup)
	wg.Add(len(inputs))
	for i, input := range inputs {
		go func(i int, input *Tensor) {
			output := NewTensor(input.shape)
			for j := 0; j < input.shape.Elements(); j++ {
				x := math.Max(input.rawData[j], 0)
				output.rawData[j] = x
			}
			outputs[i] = output
			wg.Done()
		}(i, input)
	}
	wg.Wait()
	return outputs
}

func (r *relu) Forward(inputs []*Tensor) []*Tensor {
	outputs := make([]*Tensor, len(inputs))
	r.mask = make([][]bool, len(inputs))
	wg := new(sync.WaitGroup)
	wg.Add(len(inputs))
	for i, input := range inputs {
		go func(i int, input *Tensor) {
			r.mask[i] = make([]bool, input.shape.Elements())
			output := NewTensor(input.shape)
			for j := 0; j < input.shape.Elements(); j++ {
				x := math.Max(input.rawData[j], 0)
				r.mask[i][j] = x <= 0
				output.rawData[j] = x
			}
			outputs[i] = output
			wg.Done()
		}(i, input)
	}
	wg.Wait()
	return outputs
}

func (r *relu) Backward(douts []*Tensor) []*Tensor {
	d := make([]*Tensor, len(douts))
	wg := new(sync.WaitGroup)
	wg.Add(len(douts))
	for i, dout := range douts {
		go func(i int, dout *Tensor) {
			d[i] = dout.Clone()
			for j := 0; j < d[i].shape.Elements(); j++ {
				if r.mask[i][j] {
					d[i].rawData[j] = 0
				}
			}
			wg.Done()
		}(i, dout)
	}
	wg.Wait()
	return d
}

func (r *relu) InputShape() Shape {
	return r.inputShape
}

func (r *relu) OutputShape() Shape {
	return r.outputShape
}

func (r *relu) Params() []*Tensor {
	return nil
}

func (r *relu) Update() {}

type sigmoid struct {
	inputShape  Shape
	outputShape Shape
	outputs     []*Tensor
}

// Sigmoid is an activation function layer.
func Sigmoid() Layer {
	return &sigmoid{}
}

func (s *sigmoid) Init(inputShape Shape, _ OptimizerFactory) error {
	s.inputShape = inputShape
	s.outputShape = inputShape
	return nil
}

func (s *sigmoid) Call(inputs []*Tensor) []*Tensor {
	outputs := make([]*Tensor, len(inputs))
	wg := new(sync.WaitGroup)
	wg.Add(len(inputs))
	for i, input := range inputs {
		go func(i int, input *Tensor) {
			outputs[i] = input.BroadCast(func(f float64) float64 {
				return 1 / (1 + math.Exp(-f))
			})
			wg.Done()
		}(i, input)
	}
	wg.Wait()
	return outputs
}

func (s *sigmoid) Forward(inputs []*Tensor) []*Tensor {
	s.outputs = make([]*Tensor, len(inputs))
	wg := new(sync.WaitGroup)
	wg.Add(len(inputs))
	for i, input := range inputs {
		go func(i int, input *Tensor) {
			s.outputs[i] = input.BroadCast(func(f float64) float64 {
				return 1 / (1 + math.Exp(-f))
			})
			wg.Done()
		}(i, input)
	}
	wg.Wait()
	return s.outputs
}

func (s *sigmoid) Backward(douts []*Tensor) []*Tensor {
	d := make([]*Tensor, len(douts))
	wg := new(sync.WaitGroup)
	wg.Add(len(douts))
	for i, dout := range douts {
		go func(i int, dout *Tensor) {
			d[i] = s.outputs[i].MulBroadCast(-1).AddBroadCast(1).MulTensor(s.outputs[i]).MulTensor(dout)
			wg.Done()
		}(i, dout)
	}
	wg.Wait()
	return d
}

func (s *sigmoid) InputShape() Shape {
	return s.inputShape
}

func (s *sigmoid) OutputShape() Shape {
	return s.outputShape
}

func (s *sigmoid) Params() []*Tensor {
	return nil
}

func (s *sigmoid) Update() {}

type softmax struct {
	inputShape  Shape
	outputShape Shape
	outputs     []*Tensor
}

// Sofmax is an activation function layer.
func Softmax() Layer {
	return &softmax{}
}

func (s *softmax) Init(inputShape Shape, _ OptimizerFactory) error {
	if inputShape.Rank() != 1 {
		return fmt.Errorf("invalid rank %v", inputShape.Rank())
	}

	s.inputShape = inputShape
	s.outputShape = inputShape
	return nil
}

func (s *softmax) Call(inputs []*Tensor) []*Tensor {
	outputs := make([]*Tensor, len(inputs))
	wg := new(sync.WaitGroup)
	wg.Add(len(inputs))
	for i, input := range inputs {
		go func(i int, input *Tensor) {
			max := input.Max()
			exp := input.SubBroadCast(max).Exp()
			sum := exp.Sum()
			outputs[i] = exp.BroadCast(func(f float64) float64 {
				return f / sum
			})
			wg.Done()
		}(i, input)
	}
	wg.Wait()
	return outputs
}

func (s *softmax) Forward(inputs []*Tensor) []*Tensor {
	outputs := make([]*Tensor, len(inputs))
	wg := new(sync.WaitGroup)
	wg.Add(len(inputs))
	for i, input := range inputs {
		go func(i int, input *Tensor) {
			max := input.Max()
			exp := input.SubBroadCast(max).Exp()
			sum := exp.Sum()
			outputs[i] = exp.BroadCast(func(f float64) float64 {
				return f / sum
			})
			wg.Done()
		}(i, input)
	}
	wg.Wait()
	s.outputs = outputs

	return outputs
}

func (s *softmax) Backward(douts []*Tensor) []*Tensor {
	wg := new(sync.WaitGroup)
	wg.Add(len(s.outputs))
	for i, output := range s.outputs {
		go func(i int, output *Tensor) {
			douts[i] = douts[i].MulTensor(output).AddTensor(output)
			wg.Done()
		}(i, output)
	}
	wg.Wait()
	return douts
}

func (s *softmax) InputShape() Shape {
	return s.inputShape
}

func (s *softmax) OutputShape() Shape {
	return s.outputShape
}

func (s *softmax) Params() []*Tensor {
	return nil
}

func (s *softmax) Update() {}
