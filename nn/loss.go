package nn

import "sync"

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
	wg := new(sync.WaitGroup)
	wg.Add(len(t))
	mutex := new(sync.Mutex)
	for i := 0; i < len(t); i++ {
		go func(i int) {
			d := - y[i].AddBroadCast(delta).Log().MulTensor(t[i]).Sum()
			mutex.Lock()
			sum += d
			mutex.Unlock()
			wg.Done()
		}(i)
	}
	wg.Wait()
	return sum / float64(len(t))
}

func (c *CrossEntropyError) Forward(y, t []*Tensor) float64 {
	const delta = 1e-7
	c.y = make([]*Tensor, len(y))
	c.t = make([]*Tensor, len(t))
	sum := 0.0
	wg := new(sync.WaitGroup)
	wg.Add(len(t))
	mutex := new(sync.Mutex)
	for i := 0; i < len(t); i++ {
		go func(i int) {
			c.y[i] = y[i].Clone()
			c.t[i] = t[i].Clone()
			d := - y[i].AddBroadCast(delta).Log().MulTensor(t[i]).Sum()
			mutex.Lock()
			sum += d
			mutex.Unlock()
			wg.Done()
		}(i)
	}
	wg.Wait()
	return sum / float64(len(t))
}

func (c *CrossEntropyError) Backward() []*Tensor {
	d := make([]*Tensor, len(c.y))
	wg := new(sync.WaitGroup)
	wg.Add(len(c.y))
	for i := 0; i < len(c.y); i++ {
		go func(i int) {
			d[i] = c.t[i].DivTensor(c.y[i]).MulBroadCast(-1)
			wg.Done()
		}(i)
	}
	wg.Wait()
	return d
}
