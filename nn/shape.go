package nn

type Shape []int

func (s Shape) RawIndex(at Shape) int {
	if s.Rank() != at.Rank() {
		panic("invalid rank")
	}

	index := 0
	a := 1
	for i, x := range at {
		if x >= s[i] {
			panic("index out of range")
		}

		index += x * a
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
