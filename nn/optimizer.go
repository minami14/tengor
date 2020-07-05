package nn

type Optimizer interface {
	Update(params, grads *Tensor) *Tensor
}

type OptimizerFactory interface {
	Create(Shape) Optimizer
}

type sgd struct {
	lr float64
}

func (s *sgd) Update(params, grads *Tensor) *Tensor {
	params = params.SubTensor(grads.MulBroadCast(s.lr))
	return params
}

type SGDFactory struct {
	LearningRate float64
}

func (s *SGDFactory) Create(_ Shape) Optimizer {
	return &sgd{lr: s.LearningRate}
}

type momentumSGD struct {
	lr       float64
	momentum float64
	velocity *Tensor
}

func (m *momentumSGD) Update(params, grads *Tensor) *Tensor {
	m.velocity = m.velocity.MulBroadCast(m.momentum).SubTensor(grads.MulBroadCast(m.lr))
	params = params.AddTensor(m.velocity)
	return params
}

type MomentumSGDFactory struct {
	LearningRate float64
	Momentum     float64
}

func (m *MomentumSGDFactory) Create(shape Shape) Optimizer {
	return &momentumSGD{
		lr:       m.LearningRate,
		momentum: m.Momentum,
		velocity: NewTensor(shape),
	}
}
