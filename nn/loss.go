package nn

type Loss interface {
	Call(y, t []*Tensor) float64
	Forward(y, t []*Tensor) float64
	Backward() []*Tensor
}

type CrossEntropyError struct {
	y []*Tensor
	t []*Tensor
}

func (c *CrossEntropyError) Call(y, t []*Tensor) float64 {
	const delta = 1e-7
	sum := 0.0
	for i := 0; i < len(t); i++ {
		sum += - y[i].AddBroadCast(delta).Log().MulTensor(t[i]).Sum()
	}
	return sum / float64(len(t))
}

func (c *CrossEntropyError) Forward(y, t []*Tensor) float64 {
	const delta = 1e-7
	c.y = make([]*Tensor, len(y))
	c.t = make([]*Tensor, len(t))
	sum := 0.0
	for i := 0; i < len(t); i++ {
		c.y[i] = y[i].Clone()
		c.t[i] = t[i].Clone()
		sum += - y[i].AddBroadCast(delta).Log().MulTensor(t[i]).Sum()
	}
	return sum / float64(len(t))
}

func (c *CrossEntropyError) Backward() []*Tensor {
	d := make([]*Tensor, len(c.y))
	for i := 0; i < len(c.y); i++ {
		d[i] = c.t[i].DivTensor(c.y[i]).MulBroadCast(-1)
	}
	return d
}
