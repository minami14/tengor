package nn

type Loss interface {
	Call(y, t *Tensor) float64
}

type CrossEntropyError struct{}

func (c CrossEntropyError) Call(y, t *Tensor) float64 {
	const delta = 1e-7
	return - y.AddBroadCast(delta).Log().MulTensor(t).Sum()
}
