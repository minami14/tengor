package nn

import (
	"fmt"
	"math/rand"
	"sync"
)

type Layer interface {
	InputShape() Shape
	OutputShape() Shape
	Init(inputShape Shape, factory OptimizerFactory) error
	Call(inputs []*Tensor) []*Tensor
	Forward(inputs []*Tensor) []*Tensor
	Backward(douts []*Tensor) []*Tensor
	Params() []*Tensor
	Update()
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

func (b *BaseLayer) Update() {}

type Input struct {
	BaseLayer
}

func (i *Input) Init(inputShape Shape, _ OptimizerFactory) error {
	i.inputShape = inputShape
	i.outputShape = inputShape
	return nil
}

func (i *Input) Call(inputs []*Tensor) []*Tensor {
	return inputs
}

func (i *Input) Forward(inputs []*Tensor) []*Tensor {
	return inputs
}

func (i *Input) Backward(douts []*Tensor) []*Tensor {
	return douts
}

type Dense struct {
	Units  int
	weight *Tensor
	bias   *Tensor
	inputs []*Tensor
	dw     []*Tensor
	db     []*Tensor
	OptW   Optimizer
	OptB   Optimizer
	BaseLayer
}

func (d *Dense) Init(inputShape Shape, factory OptimizerFactory) error {
	if inputShape.Rank() != 1 {
		return fmt.Errorf("invalid rank %v", inputShape.Rank())
	}

	d.inputShape = inputShape
	d.outputShape = Shape{d.Units}
	wShape := Shape{inputShape[0], d.Units}
	d.weight = NewTensor(wShape)
	d.weight = d.weight.BroadCast(func(_ float64) float64 {
		return rand.Float64() * 0.01
	})
	d.bias = NewTensor(d.outputShape)
	d.OptW = factory.Create(wShape)
	d.OptB = factory.Create(d.outputShape)
	return nil
}

func (d *Dense) Call(inputs []*Tensor) []*Tensor {
	outputs := make([]*Tensor, len(inputs))
	wg := new(sync.WaitGroup)
	wg.Add(len(inputs))
	for i, input := range inputs {
		go func(i int, input *Tensor) {
			outputs[i] = input.ReShape(Shape{1, input.shape[0]}).Dot(d.weight).ReShape(d.outputShape).AddTensor(d.bias)
			wg.Done()
		}(i, input)
	}
	wg.Wait()
	return outputs
}

func (d *Dense) Forward(inputs []*Tensor) []*Tensor {
	d.inputs = make([]*Tensor, len(inputs))
	outputs := make([]*Tensor, len(inputs))
	wg := new(sync.WaitGroup)
	wg.Add(len(inputs))
	for i, input := range inputs {
		go func(i int, input *Tensor) {
			d.inputs[i] = input
			outputs[i] = input.ReShape(Shape{1, input.shape[0]}).Dot(d.weight).ReShape(d.outputShape).AddTensor(d.bias)
			wg.Done()
		}(i, input)
	}
	wg.Wait()
	return outputs
}

func (d *Dense) Backward(douts []*Tensor) []*Tensor {
	d.dw = make([]*Tensor, len(douts))
	d.db = make([]*Tensor, len(douts))
	dx := make([]*Tensor, len(douts))
	wg := new(sync.WaitGroup)
	wg.Add(len(douts))
	for i, dout := range douts {
		go func(i int, dout *Tensor) {
			d.db[i] = dout.Clone()
			dout = dout.ReShape(Shape{1, dout.shape[0]})
			dx[i] = dout.Dot(d.weight.Transpose())
			dx[i] = dx[i].ReShape(Shape{dx[i].shape[1]})
			d.dw[i] = d.inputs[i].ReShape(Shape{1, d.inputs[i].shape[0]}).Transpose().Dot(dout)
			wg.Done()
		}(i, dout)
	}
	wg.Wait()
	return dx
}

func (d *Dense) Params() []*Tensor {
	return []*Tensor{d.weight, d.bias}
}

func (d *Dense) Update() {
	dw := NewTensor(d.dw[0].shape)
	db := NewTensor(d.db[0].shape)
	for i := 0; i < len(d.dw); i++ {
		dw = dw.AddTensor(d.dw[i])
		db = db.AddTensor(d.db[i])
	}
	dw = dw.DivBroadCast(float64(len(d.dw)))
	db = db.DivBroadCast(float64(len(d.db)))
	d.weight = d.OptW.Update(d.weight, dw)
	d.bias = d.OptB.Update(d.bias, db)
}

type Flatten struct {
	BaseLayer
}

func (f *Flatten) Init(inputShape Shape, _ OptimizerFactory) error {
	f.inputShape = inputShape
	f.outputShape = Shape{inputShape.Elements()}
	return nil
}

func (f *Flatten) Call(inputs []*Tensor) []*Tensor {
	outputs := make([]*Tensor, len(inputs))
	for i, input := range inputs {
		outputs[i] = input.Clone()
		outputs[i].shape = f.outputShape.Clone()
	}
	return outputs
}

func (f *Flatten) Forward(inputs []*Tensor) []*Tensor {
	return f.Call(inputs)
}

func (f *Flatten) Backward(douts []*Tensor) []*Tensor {
	return douts
}

type Dropout struct {
	Rate float64
	mask [][]bool
	BaseLayer
}

func (d *Dropout) Init(inputShape Shape, _ OptimizerFactory) error {
	d.inputShape = inputShape
	d.outputShape = inputShape
	return nil
}

func (d *Dropout) Call(inputs []*Tensor) []*Tensor {
	return inputs
}

func (d *Dropout) Forward(inputs []*Tensor) []*Tensor {
	d.mask = make([][]bool, len(inputs))
	units := inputs[0].shape.Elements()
	active := int(float64(units) * (1 - d.Rate))
	for i, input := range inputs {
		mask := make([]bool, units)
		for n := 0; n < active; {
			index := rand.Intn(units)
			if mask[index] {
				continue
			}
			input.rawData[index] = 0
			mask[index] = true
			n++
		}
		d.mask[i] = mask
	}
	return inputs
}

func (d *Dropout) Backward(douts []*Tensor) []*Tensor {
	for i, dout := range douts {
		for j, drop := range d.mask[i] {
			if drop {
				dout.rawData[j] = 0
			}
		}
	}
	return douts
}
