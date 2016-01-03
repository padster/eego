// Simple implementation to find linear regression for a given set of (x, y) values
// of gradient descent rather than a closed formula.

package ml

import (
	"fmt"
)

// Polynomial coefficients, state[0] + state[1] * x
type GDLRState [2]float64

type GradDescLinReg struct {
	state GDLRState
	alpha float64
}

// State for performing linear regression by gradient descent.
func NewGradDescLinReg(alpha float64) *GradDescLinReg {
	return &GradDescLinReg{
		[...]float64{0., 0.},
		alpha,
	}
}

// Train performs gradient descent on the given data to find the linear regression.
func (ml *GradDescLinReg) Train(inputs []float64, training []float64) GDLRState {
	if len(inputs) != len(training) {
		panic("Inputs to train must be the same size")
	}

	ml.state[0], ml.state[1] = 0.0, 0.0
	
	iterations := 0
	updateDistSq := 1.0

	for updateDistSq > 1e-15 {
		if iterations % 1000 == 0 {
			fmt.Printf("#%d\t:\t%f - %f\n", iterations, ml.state[0], ml.state[1])
		}
		if iterations > 10000 {
			panic("No convergence")
		}
		iterations++

		nextState := [...]float64{0., 0.}
		nextState[0] = ml.state[0] - ml.alpha * ml.meanDist(inputs, training)
		nextState[1] = ml.state[1] - ml.alpha * ml.meanScaledDist(inputs, training)
		updateDistSq = DistSq(ml.state[:], nextState[:])
		ml.state = nextState
	}
	return ml.state
}

func (ml *GradDescLinReg) meanDist(inputs []float64, training []float64) float64 {
	md := 0.0
	for i, _ := range inputs {
		md += ml.estimate(inputs[i]) - training[i]
	}
	return md / float64(len(inputs))
}

func (ml *GradDescLinReg) meanScaledDist(inputs []float64, training []float64) float64 {
	msd := 0.0
	for i, _ := range inputs {
		msd += (ml.estimate(inputs[i]) - training[i]) * inputs[i]
	}
	return msd / float64(len(inputs))
}

func (ml *GradDescLinReg) estimate(input float64) float64 {
	return ml.state[0] + ml.state[1] * input
}
