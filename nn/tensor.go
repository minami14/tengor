package nn

import (
	"math"
)

// Tensor is an algebraic object that describes a relationship between sets of algebraic objects related to a vector space.
type Tensor struct {
	shape   Shape
	rawData []float64
}

// NewTensor creates an instance of tensor.
func NewTensor(shape Shape) *Tensor {
	return &Tensor{
		shape:   shape.Clone(),
		rawData: make([]float64, shape.Elements()),
	}
}

// TensorFromSlice creates an instance of tensor initialized with a given data.
func TensorFromSlice(shape Shape, p []float64) *Tensor {
	if shape.Elements() != len(p) {
		panic("invalid length")
	}

	tensor := NewTensor(shape)
	copy(tensor.rawData, p)

	return tensor
}

// ReShape reshapes a tensor.
func (t *Tensor) ReShape(shape Shape) *Tensor {
	if t.shape.Elements() != shape.Elements() {
		panic("invalid shape")
	}

	res := t.Clone()
	res.shape = shape
	return res
}

// Clone clones a tensor.
func (t *Tensor) Clone() *Tensor {
	clone := NewTensor(t.shape.Clone())
	copy(clone.rawData, t.rawData)
	return clone
}

// Shape is shape of a tensor.
func (t *Tensor) Shape() Shape {
	return t.shape.Clone()
}

// Rank is rank of a tensor.
func (t *Tensor) Rank() int {
	return t.shape.Rank()
}

// Get gets a value.
func (t *Tensor) Get(at Shape) float64 {
	return t.rawData[t.shape.RawIndex(at)]
}

// Set sets a value.
func (t *Tensor) Set(a float64, at Shape) {
	t.rawData[t.shape.RawIndex(at)] = a
}

// BroadCast creates a tensor of the return value that inputs all the elements into the passed function.
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

// AddBroadCast adds a value to all elements.
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

// SubBroadCast subtracts a value​from all elements.
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

// MulBroadCast multiplies all elements by a value.
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

// DivBroadCast divides all elements by a value.
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

// AddTensor adds a tensor.
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

// SubTensor subtracts a tensor.
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

// MulTensor multiplies by a tensor.
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

// DivTensor divides by a tensor.
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

// Dot is a dot product of tensor.
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

// Sum is sum of all elements.
func (t *Tensor) Sum() float64 {
	var res float64
	for _, d := range t.rawData {
		res += d
	}

	return res
}

// Exp is exp of a tensor.
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

// Log is log of a tensor.
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

// Transpose transpose tensor.
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

// Max is maximum value of a tensor.
func (t *Tensor) Max() float64 {
	max := t.rawData[0]
	for _, x := range t.rawData {
		if x > max {
			max = x
		}
	}
	return max
}

// MaxIndex is a index of a maximum value.
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
