package nn

import (
	"fmt"
	"math"
)

type ReLU struct {
	BaseLayer
	mask [][]bool
}

func (r *ReLU) Init(inputShape Shape) error {
	r.inputShape = inputShape
	r.outputShape = inputShape
	return nil
}

func (r *ReLU) Call(inputs []*Tensor) []*Tensor {
	outputs := make([]*Tensor, len(inputs))
	for i, input := range inputs {
		output := NewTensor(input.shape)
		for j := 0; j < input.shape.Elements(); j++ {
			x := math.Max(input.rawData[j], 0)
			output.rawData[j] = x
		}
		outputs[i] = output
	}

	return outputs
}

func (r *ReLU) Forward(inputs []*Tensor) []*Tensor {
	outputs := make([]*Tensor, len(inputs))
	r.mask = make([][]bool, len(inputs))
	for i, input := range inputs {
		r.mask[i] = make([]bool, input.shape.Elements())
		output := NewTensor(input.shape)
		for j := 0; j < input.shape.Elements(); j++ {
			x := math.Max(input.rawData[j], 0)
			r.mask[i][j] = x <= 0
			output.rawData[j] = x
		}
		outputs[i] = output
	}

	return outputs
}

func (r *ReLU) Backward(douts []*Tensor) []*Tensor {
	d := make([]*Tensor, len(douts))
	for i, dout := range douts {
		d[i] = dout.Clone()
		for j := 0; j < d[i].shape.Elements(); j++ {
			if r.mask[i][j] {
				d[i].rawData[j] = 0
			}
		}
	}
	return d
}

type Sigmoid struct {
	BaseLayer
	outputs []*Tensor
}

func (s *Sigmoid) Init(inputShape Shape) error {
	s.inputShape = inputShape
	s.outputShape = inputShape
	return nil
}

func (s *Sigmoid) Call(inputs []*Tensor) []*Tensor {
	outputs := make([]*Tensor, len(inputs))
	for i, input := range inputs {
		outputs[i] = input.BroadCast(func(f float64) float64 {
			return 1 / (1 + math.Exp(-f))
		})
	}

	return outputs
}

func (s *Sigmoid) Forward(inputs []*Tensor) []*Tensor {
	s.outputs = make([]*Tensor, len(inputs))
	for i, input := range inputs {
		s.outputs[i] = input.BroadCast(func(f float64) float64 {
			return 1 / (1 + math.Exp(-f))
		})
	}

	return s.outputs
}

func (s *Sigmoid) Backward(douts []*Tensor) []*Tensor {
	d := make([]*Tensor, len(douts))
	for i, dout := range douts {
		d[i] = s.outputs[i].MulBroadCast(-1).AddBroadCast(1).MulTensor(s.outputs[i]).MulTensor(dout)
	}
	return d
}

type Softmax struct {
	BaseLayer
	outputs []*Tensor
}

func (s *Softmax) Init(inputShape Shape) error {
	if inputShape.Rank() != 1 {
		return fmt.Errorf("invalid rank %v", inputShape.Rank())
	}

	s.inputShape = inputShape
	s.outputShape = inputShape
	return nil
}

func (s *Softmax) Call(inputs []*Tensor) []*Tensor {
	outputs := make([]*Tensor, len(inputs))
	for i, input := range inputs {
		max := input.Max()
		exp := input.SubBroadCast(max).Exp()
		sum := exp.Sum()
		outputs[i] = exp.BroadCast(func(f float64) float64 {
			return f / sum
		})
	}

	return outputs
}

func (s *Softmax) Forward(inputs []*Tensor) []*Tensor {
	outputs := make([]*Tensor, len(inputs))
	for i, input := range inputs {
		max := input.Max()
		exp := input.SubBroadCast(max).Exp()
		sum := exp.Sum()
		outputs[i] = exp.BroadCast(func(f float64) float64 {
			return f / sum
		})
	}
	s.outputs = outputs

	return outputs
}

func (s *Softmax) Backward(douts []*Tensor) []*Tensor {
	for i, output := range s.outputs {
		douts[i] = douts[i].MulTensor(output).AddTensor(output)
	}
	return douts
}

type Lambda struct {
	Function        func(*Tensor) *Tensor
	CalcOutputShape func(inputShape Shape) Shape
	BaseLayer
}

func (l *Lambda) Init(inputShape Shape) error {
	l.inputShape = inputShape
	l.outputShape = l.CalcOutputShape(inputShape)
	return nil
}

func (l *Lambda) Call(inputs []*Tensor) []*Tensor {
	outputs := make([]*Tensor, len(inputs))
	for i, input := range inputs {
		outputs[i] = l.Function(input)
	}
	return outputs
}

func (l *Lambda) Forward(inputs []*Tensor) []*Tensor {
	return l.Call(inputs)
}

func (l *Lambda) Backward(douts []*Tensor) []*Tensor {
	return douts
}
