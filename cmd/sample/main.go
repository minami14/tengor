package main

import (
	"fmt"
	"log"

	"github.com/minami14/tengor/dataset/mnist"
	"github.com/minami14/tengor/nn"
)

const (
	epochs    = 10
	batchSize = 100
	lr        = 0.1
	momentum  = 0.9
)

func main() {
	xTrain, yTrain, xTest, yTest, err := mnist.Load()
	if err != nil {
		log.Fatal(err)
	}

	inputShape := nn.Shape{28, 28}
	model := nn.NewSequential(inputShape)
	model.AddLayer(nn.Flatten())
	model.AddLayer(nn.Dense(64))
	model.AddLayer(nn.ReLU())
	model.AddLayer(nn.Dropout(0.5))
	model.AddLayer(nn.Dense(10))
	model.AddLayer(nn.Softmax())
	if err := model.Build(nn.CrossEntropyError(), nn.MomentumSGD(lr, momentum)); err != nil {
		log.Fatal(err)
	}
	fmt.Println(model.Summary())

	model.Fit(xTrain, yTrain, epochs, batchSize)

	pred := model.Predict(xTest)
	loss := model.Loss(pred, yTest)
	acc := model.Accuracy(pred, yTest)
	fmt.Printf("loss: %.4f\nacc: %.4f\n", loss, acc)
}
