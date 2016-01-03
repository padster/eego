package main

import (
	"fmt"
	"github.com/padster/eego/ml"
)

func main() {
	gdlr := ml.NewGradDescLinReg(0.01)

	fit := gdlr.Train(
		[]float64{9, 5, 12},
		[]float64{2, 1, 3},
	)

	fmt.Printf("Best fit: %f + %f * x\n", fit[0], fit[1])
}
