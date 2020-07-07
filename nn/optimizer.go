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

type sgdFactory struct {
	lr float64
}

func (s *sgdFactory) Create(_ Shape) Optimizer {
	return &sgd{lr: s.lr}
}

func SGD(lr float64) OptimizerFactory {
	return &sgdFactory{lr}
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

type momentumSGDFactory struct {
	lr       float64
	momentum float64
}

func (m *momentumSGDFactory) Create(shape Shape) Optimizer {
	return &momentumSGD{
		lr:       m.lr,
		momentum: m.momentum,
		velocity: NewTensor(shape),
	}
}

func MomentumSGD(lr, momentum float64) OptimizerFactory {
	if momentum == 0 {
		return SGD(lr)
	}

	return &momentumSGDFactory{
		lr:       lr,
		momentum: momentum,
	}
}
