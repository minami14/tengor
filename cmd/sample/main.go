package main

import (
	"fmt"
	"github.com/minami14/tengor/nn"
	"log"
)

func main() {
	inputShape := nn.Shape{28, 28}
	model := nn.NewSequential(inputShape)
	model.AddLayer(&nn.Flatten{})
	model.AddLayer(&nn.Dense{Units: 64})
	model.AddLayer(&nn.ReLU{})
	model.AddLayer(&nn.Dense{Units: 10})
	model.AddLayer(&nn.Softmax{})
	if err := model.Build(nn.CrossEntropyError{}); err != nil {
		log.Fatal(err)
	}
	fmt.Println(model.Summary())
}
