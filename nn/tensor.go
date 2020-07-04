package nn

import (
	"math"
)

type Tensor struct {
	shape   Shape
	rawData []float64
}

func NewTensor(shape Shape) *Tensor {
	return &Tensor{
		shape:   shape.Clone(),
		rawData: make([]float64, shape.Elements()),
	}
}

func TensorFromSlice(shape Shape, p []float64) *Tensor {
	if shape.Elements() != len(p) {
		panic("invalid length")
	}

	tensor := NewTensor(shape)
	copy(tensor.rawData, p)

	return tensor
}

func (t *Tensor) ReShape(shape Shape) *Tensor {
	if t.shape.Elements() != shape.Elements() {
		panic("invalid shape")
	}

	res := t.Clone()
	res.shape = shape
	return res
}

func (t *Tensor) Clone() *Tensor {
	clone := NewTensor(t.shape.Clone())
	copy(clone.rawData, t.rawData)
	return clone
}

func (t *Tensor) Shape() Shape {
	return t.shape.Clone()
}

func (t *Tensor) Rank() int {
	return t.shape.Rank()
}

func (t *Tensor) Get(at Shape) float64 {
	return t.rawData[t.shape.RawIndex(at)]
}

func (t *Tensor) Set(a float64, at Shape) {
	t.rawData[t.shape.RawIndex(at)] = a
}

func (t *Tensor) BroadCast(f func(float64) float64) *Tensor {
	res := &Tensor{
		shape:   t.Shape(),
		rawData: make([]float64, len(t.rawData)),
	}
	for i, d := range t.rawData {
		res.rawData[i] = f(d)
	}

	return res
}

func (t *Tensor) AddBroadCast(a float64) *Tensor {
	res := &Tensor{
		shape:   t.Shape(),
		rawData: make([]float64, len(t.rawData)),
	}
	for i, d := range t.rawData {
		res.rawData[i] = d + a
	}

	return res
}

func (t *Tensor) SubBroadCast(a float64) *Tensor {
	res := &Tensor{
		shape:   t.Shape(),
		rawData: make([]float64, len(t.rawData)),
	}
	for i, d := range t.rawData {
		res.rawData[i] = d - a
	}

	return res
}

func (t *Tensor) MulBroadCast(a float64) *Tensor {
	res := &Tensor{
		shape:   t.Shape(),
		rawData: make([]float64, len(t.rawData)),
	}
	for i, d := range t.rawData {
		res.rawData[i] = d * a
	}

	return res
}

func (t *Tensor) DivBroadCast(a float64) *Tensor {
	res := &Tensor{
		shape:   t.Shape(),
		rawData: make([]float64, len(t.rawData)),
	}
	for i, d := range t.rawData {
		res.rawData[i] = d / a
	}

	return res
}

func (t *Tensor) AddTensor(tensor *Tensor) *Tensor {
	if !t.shape.Equal(tensor.shape) {
		panic("invalid shape")
	}

	res := &Tensor{
		shape:   t.Shape(),
		rawData: make([]float64, len(t.rawData)),
	}

	for i := 0; i < len(t.rawData); i++ {
		res.rawData[i] = t.rawData[i] + tensor.rawData[i]
	}

	return res
}

func (t *Tensor) SubTensor(tensor *Tensor) *Tensor {
	if !t.shape.Equal(tensor.shape) {
		panic("invalid shape")
	}

	res := &Tensor{
		shape:   t.Shape(),
		rawData: make([]float64, len(t.rawData)),
	}

	for i := 0; i < len(t.rawData); i++ {
		res.rawData[i] = t.rawData[i] - tensor.rawData[i]
	}

	return res
}

func (t *Tensor) MulTensor(tensor *Tensor) *Tensor {
	if !t.shape.Equal(tensor.shape) {
		panic("invalid shape")
	}

	res := &Tensor{
		shape:   t.Shape(),
		rawData: make([]float64, len(t.rawData)),
	}

	for i := 0; i < len(t.rawData); i++ {
		res.rawData[i] = t.rawData[i] * tensor.rawData[i]
	}

	return res
}

func (t *Tensor) DivTensor(tensor *Tensor) *Tensor {
	if !t.shape.Equal(tensor.shape) {
		panic("invalid shape")
	}

	res := &Tensor{
		shape:   t.Shape(),
		rawData: make([]float64, len(t.rawData)),
	}

	for i := 0; i < len(t.rawData); i++ {
		res.rawData[i] = t.rawData[i] / tensor.rawData[i]
	}

	return res
}

func (t *Tensor) Dot(tensor *Tensor) *Tensor {
	t1, t2 := t, tensor
	if t1.Rank() != 2 || t2.Rank() != 2 || t1.shape[1] != t2.shape[0] {
		panic("invalid rank")
	}

	res := NewTensor(Shape{t1.shape[0], t2.shape[1]})
	for i := 0; i < t1.shape[0]; i++ {
		for j := 0; j < t2.shape[1]; j++ {
			for k := 0; k < t2.shape[0]; k++ {
				val := res.Get(Shape{i, j}) + t1.Get(Shape{i, k})*t2.Get(Shape{k, j})
				res.Set(val, Shape{i, j})
			}
		}
	}

	return res
}

func (t *Tensor) Sum() float64 {
	var res float64
	for _, d := range t.rawData {
		res += d
	}

	return res
}

func (t *Tensor) Exp() *Tensor {
	res := &Tensor{
		shape:   t.Shape(),
		rawData: make([]float64, len(t.rawData)),
	}
	for i, d := range t.rawData {
		res.rawData[i] = math.Exp(d)
	}

	return res
}

func (t *Tensor) Log() *Tensor {
	res := &Tensor{
		shape:   t.Shape(),
		rawData: make([]float64, len(t.rawData)),
	}
	for i, d := range t.rawData {
		res.rawData[i] = math.Log(d)
	}

	return res
}

func (t *Tensor) Transpose() *Tensor {
	if t.Rank() != 2 {
		panic("invalid rank")
	}

	res := NewTensor(Shape{t.shape[1], t.shape[0]})
	for i := 0; i < t.shape[0]; i++ {
		for j := 0; j < t.shape[1]; j++ {
			res.Set(t.Get(Shape{i, j}), Shape{j, i})
		}
	}
	return res
}

func (t *Tensor) Max() float64 {
	max := t.rawData[0]
	for _, x := range t.rawData {
		if x > max {
			max = x
		}
	}
	return max
}

func (t *Tensor) MaxIndex() int {
	index := 0
	max := -1.0
	for i := 0; i < t.shape.Elements(); i++ {
		if max < t.rawData[i] {
			max = t.rawData[i]
			index = i
		}
	}
	return index
}
