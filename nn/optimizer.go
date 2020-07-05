package nn

type Optimizer interface {
	Update(params, grads *Tensor) *Tensor
}

type OptimizerFactory interface {
	Create(Shape) Optimizer
}

type SGD struct {
	LearningRate float64
}

func (s *SGD) Update(params, grads *Tensor) *Tensor {
	params = params.SubTensor(grads.MulBroadCast(s.LearningRate))
	return params
}

type SGDFactory struct {
	LearningRate float64
}

func (s *SGDFactory) Create(_ Shape) Optimizer {
	return &SGD{LearningRate: s.LearningRate}
}
