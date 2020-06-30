package nn

import (
	"fmt"
	"math"
)

type ReLU struct {
	BaseLayer
}

func (r *ReLU) Init(inputShape Shape) error {
	r.inputShape = inputShape
	r.outputShape = inputShape
	return nil
}

func (r *ReLU) Call(input *Tensor) *Tensor {
	return input.BroadCast(func(f float64) float64 {
		return math.Max(f, 0)
	})
}

type Sigmoid struct {
	BaseLayer
}

func (s *Sigmoid) Init(inputShape Shape) error {
	s.inputShape = inputShape
	s.outputShape = inputShape
	return nil
}

func (s *Sigmoid) Call(input *Tensor) *Tensor {
	return input.BroadCast(func(f float64) float64 {
		return 1 / (1 + math.Exp(-f))
	})
}

type Softmax struct {
	BaseLayer
}

func (s *Softmax) Init(inputShape Shape) error {
	if inputShape.Rank() != 1 {
		return fmt.Errorf("invalid rank %v", inputShape.Rank())
	}

	s.inputShape = inputShape
	s.outputShape = inputShape
	return nil
}

func (s *Softmax) Call(input *Tensor) *Tensor {
	exp := input.Exp()
	sum := exp.Sum()
	output := exp.BroadCast(func(f float64) float64 {
		return f / sum
	})

	return output
}

type Lambda struct {
	BaseLayer
	Function        func(*Tensor) *Tensor
	CalcOutputShape func(inputShape Shape) Shape
}

func (l *Lambda) Init(inputShape Shape) error {
	l.inputShape = inputShape
	l.outputShape = l.CalcOutputShape(inputShape)
	return nil
}

func (l *Lambda) Call(input *Tensor) *Tensor {
	return l.Function(input)
}
