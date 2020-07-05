package nn

type Optimizer interface {
	Update(params, grads *Tensor) *Tensor
}

type SGD struct {
	LearningRate float64
}

func (s *SGD) Update(params, grads *Tensor) *Tensor {
	params = params.SubTensor(grads.MulBroadCast(s.LearningRate))
	return params
}
