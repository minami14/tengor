package nn

type Shape []int

func (s Shape) RawIndex(at Shape) int {
	if s.Rank() != at.Rank() {
		panic("invalid rank")
	}

	index := at[len(at)-1]
	a := s[len(s)-1]
	for i := len(at) - 2; i >= 0; i-- {
		if at[i] > s[i] {
			panic("index out of range")
		}

		index += at[i] * a
		a *= s[i]
	}

	return index
}

func (s Shape) Clone() Shape {
	clone := make(Shape, len(s))
	for i, d := range s {
		clone[i] = d
	}
	return clone
}

func (s Shape) Rank() int {
	return len(s)
}

func (s Shape) Elements() int {
	e := 1
	for _, i := range s {
		e *= i
	}

	return e
}

func (s Shape) Equal(shape Shape) bool {
	if len(s) != len(shape) {
		return false
	}

	for i := 0; i < len(s); i++ {
		if s[i] != shape[i] {
			return false
		}
	}

	return true
}
