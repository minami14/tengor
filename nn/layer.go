package nn

import (
	"fmt"
	"math/rand"
)

type Layer interface {
	InputShape() Shape
	OutputShape() Shape
	Init(inputShape Shape) error
	Call(input *Tensor) *Tensor
	Params() []*Tensor
}

type BaseLayer struct {
	inputShape  Shape
	outputShape Shape
}

func (b *BaseLayer) InputShape() Shape {
	return b.inputShape
}

func (b *BaseLayer) OutputShape() Shape {
	return b.outputShape
}

func (b *BaseLayer) Params() []*Tensor {
	return nil
}

type Input struct {
	BaseLayer
}

func (i *Input) Init(inputShape Shape) error {
	i.inputShape = inputShape
	i.outputShape = inputShape
	return nil
}

func (i *Input) Call(input *Tensor) *Tensor {
	return input
}

type Dense struct {
	BaseLayer
	weight *Tensor
	bias   *Tensor
	Units  int
}

func (d *Dense) Init(inputShape Shape) error {
	if inputShape.Rank() != 1 {
		return fmt.Errorf("invalid rank %v", inputShape.Rank())
	}

	d.inputShape = inputShape
	d.outputShape = Shape{d.Units}
	d.weight = NewTensor(Shape{inputShape[0], d.Units})
	d.weight.BroadCast(func(_ float64) float64 {
		return rand.Float64() * 0.01
	})
	d.bias = NewTensor(d.outputShape)
	return nil
}

func (d *Dense) Call(input *Tensor) *Tensor {
	return input.ReShape(Shape{1, input.shape[0]}).Dot(d.weight).ReShape(d.outputShape).AddTensor(d.bias)
}

func (d *Dense) Params() []*Tensor {
	return []*Tensor{d.weight, d.bias}
}

type Flatten struct {
	BaseLayer
}

func (f *Flatten) Init(inputShape Shape) error {
	f.inputShape = inputShape
	f.outputShape = Shape{inputShape.Elements()}
	return nil
}

func (f *Flatten) Call(input *Tensor) *Tensor {
	t := input.Clone()
	t.shape = f.outputShape
	return t
}
