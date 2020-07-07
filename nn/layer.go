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

type inputLayer struct {
	inputShape  Shape
	outputShape Shape
}

func (i *inputLayer) Init(inputShape Shape, _ OptimizerFactory) error {
	i.inputShape = inputShape
	i.outputShape = inputShape
	return nil
}

func (i *inputLayer) Call(inputs []*Tensor) []*Tensor {
	return inputs
}

func (i *inputLayer) Forward(inputs []*Tensor) []*Tensor {
	return inputs
}

func (i *inputLayer) Backward(douts []*Tensor) []*Tensor {
	return douts
}

func (i *inputLayer) InputShape() Shape {
	return i.inputShape
}

func (i *inputLayer) OutputShape() Shape {
	return i.outputShape
}

func (i *inputLayer) Params() []*Tensor {
	return nil
}

func (i *inputLayer) Update() {}

type dense struct {
	units       int
	weight      *Tensor
	bias        *Tensor
	inputs      []*Tensor
	dw          []*Tensor
	db          []*Tensor
	optW        Optimizer
	optB        Optimizer
	inputShape  Shape
	outputShape Shape
}

func Dense(units int) Layer {
	return &dense{units: units}
}

func (d *dense) Init(inputShape Shape, factory OptimizerFactory) error {
	if inputShape.Rank() != 1 {
		return fmt.Errorf("invalid rank %v", inputShape.Rank())
	}

	d.inputShape = inputShape
	d.outputShape = Shape{d.units}
	wShape := Shape{inputShape[0], d.units}
	d.weight = NewTensor(wShape)
	d.weight = d.weight.BroadCast(func(_ float64) float64 {
		return rand.Float64() * 0.01
	})
	d.bias = NewTensor(d.outputShape)
	d.optW = factory.Create(wShape)
	d.optB = factory.Create(d.outputShape)
	return nil
}

func (d *dense) Call(inputs []*Tensor) []*Tensor {
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

func (d *dense) Forward(inputs []*Tensor) []*Tensor {
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

func (d *dense) Backward(douts []*Tensor) []*Tensor {
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

func (d *dense) Params() []*Tensor {
	return []*Tensor{d.weight, d.bias}
}

func (d *dense) Update() {
	dw := NewTensor(d.dw[0].shape)
	db := NewTensor(d.db[0].shape)
	for i := 0; i < len(d.dw); i++ {
		dw = dw.AddTensor(d.dw[i])
		db = db.AddTensor(d.db[i])
	}
	dw = dw.DivBroadCast(float64(len(d.dw)))
	db = db.DivBroadCast(float64(len(d.db)))
	d.weight = d.optW.Update(d.weight, dw)
	d.bias = d.optB.Update(d.bias, db)
}

func (d *dense) InputShape() Shape {
	return d.inputShape
}

func (d *dense) OutputShape() Shape {
	return d.outputShape
}

type flatten struct {
	inputShape  Shape
	outputShape Shape
}

func Flatten() Layer {
	return &flatten{}
}

func (f *flatten) Init(inputShape Shape, _ OptimizerFactory) error {
	f.inputShape = inputShape
	f.outputShape = Shape{inputShape.Elements()}
	return nil
}

func (f *flatten) Call(inputs []*Tensor) []*Tensor {
	outputs := make([]*Tensor, len(inputs))
	for i, input := range inputs {
		outputs[i] = input.Clone()
		outputs[i].shape = f.outputShape.Clone()
	}
	return outputs
}

func (f *flatten) Forward(inputs []*Tensor) []*Tensor {
	return f.Call(inputs)
}

func (f *flatten) Backward(douts []*Tensor) []*Tensor {
	return douts
}

func (f *flatten) InputShape() Shape {
	return f.inputShape
}

func (f *flatten) OutputShape() Shape {
	return f.outputShape
}

func (f *flatten) Params() []*Tensor {
	return nil
}

func (f *flatten) Update() {}

type dropout struct {
	rate        float64
	mask        [][]bool
	inputShape  Shape
	outputShape Shape
}

func Dropout(rate float64) Layer {
	return &dropout{rate: rate}
}

func (d *dropout) Init(inputShape Shape, _ OptimizerFactory) error {
	d.inputShape = inputShape
	d.outputShape = inputShape
	return nil
}

func (d *dropout) Call(inputs []*Tensor) []*Tensor {
	return inputs
}

func (d *dropout) Forward(inputs []*Tensor) []*Tensor {
	d.mask = make([][]bool, len(inputs))
	units := inputs[0].shape.Elements()
	active := int(float64(units) * (1 - d.rate))
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

func (d *dropout) Backward(douts []*Tensor) []*Tensor {
	for i, dout := range douts {
		for j, drop := range d.mask[i] {
			if drop {
				dout.rawData[j] = 0
			}
		}
	}
	return douts
}

func (d *dropout) InputShape() Shape {
	return d.inputShape
}

func (d *dropout) OutputShape() Shape {
	return d.outputShape
}

func (d *dropout) Params() []*Tensor {
	return nil
}

func (d *dropout) Update() {}
