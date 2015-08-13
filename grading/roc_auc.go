package grading

import (
	"math"
	"sort"
)

// RocAuc returns the area under the Receiver operating characteristic (ROC) curve
// See https://en.wikipedia.org/wiki/Receiver_operating_characteristic
func RocAucScore(actual []int, predictions []float64) float64 {
	// TODO: verify that actual contains both 0s and 1s, and nothing else, and both are same size.
	fps, tps, _ := rocCurve(actual, predictions)
	return auc(fps, tps, true /* reorder */)
}

// rocCurve takes an array of [0, 1] events, plus predicted probabilities, and returns
// (fps, tps, thresholds) where:
// thresholds[i] = the different guess thresholds possible
// fps = false positive rate at each threshold
// tps = true positive rate at each threshold
func rocCurve(actual []int, predictions []float64) ([]float64, []float64, []float64) {
	fps, tps, thresh := binaryClfCurve(actual, predictions)
	n := len(fps)

	if n == 0 {
		panic("Can't find thresholds in rocCurve.")
	}

	if fps[n-1] != 0 {
		fps = append(fps, 0.0)
		tps = append(tps, 0.0)
		thresh = append(thresh, thresh[n-1]+1.0)
		n++
	}

	if fps[0] == 0 || tps[0] == 0 {
		panic("Can't score: actual data is either all false or all true.")
	}

	normFps, normTps := make([]float64, n, n), make([]float64, n, n)
	scaleFps, scaleTps := 1.0/float64(fps[0]), 1.0/float64(tps[0])
	for i := 0; i < n; i++ {
		normFps[i] = float64(fps[i]) * scaleFps
		normTps[i] = float64(tps[i]) * scaleTps
	}
	return normFps, normTps, thresh
}

// binaryClfCurve identifies the important classification thresholds, and
// calculates true and false positive counts for each.
func binaryClfCurve(actual []int, predictions []float64) ([]int, []int, []float64) {
	n := len(actual)
	fps, tps, thresh := make([]int, 0, n), make([]int, 0, n), make([]float64, 0, n)

	toSort := DualSortFI{predictions, actual}
	sort.Sort(toSort)
	actual, predictions = toSort.v2, toSort.v1

	truePos := 0
	for _, v := range actual {
		if v == 1 {
			truePos++
		}
	}
	falsePos := n - truePos

	for i := 0; i < n; i++ {
		shouldGuess := i == 0 || !fpeq(predictions[i], predictions[i-1])
		if shouldGuess {
			fps = append(fps, falsePos)
			tps = append(tps, truePos)
			thresh = append(thresh, predictions[i])
		}

		if actual[i] == 0 {
			falsePos--
		} else {
			truePos--
		}
	}

	return fps, tps, thresh
}

// Calculate area under the given curve using trapezoidal rules
func auc(xs []float64, ys []float64, reorder bool) float64 {
	if len(xs) < 2 || len(xs) != len(ys) {
		panic("auc() requires two equal length arrays of size >= 2")
	}

	toSort := DualSortFF{xs, ys}
	if reorder {
		sort.Sort(toSort)
	}
	return trapz(toSort.v2, toSort.v1)
}

// Calculate the area using the trapezium rule
func trapz(ys []float64, xs []float64) float64 {
	n := len(ys)
	ans := 0.0
	for i := 1; i < n; i++ {
		ans += (xs[i] - xs[i-1]) * (ys[i] + ys[i-1])
	}
	return ans * 0.5
}

// Sortable structure of two lists, by the first then the second
// UUUGGH, go missing templates/generics.
type DualSortFF struct {
	v1 []float64
	v2 []float64
}

func (vs DualSortFF) Len() int {
	return len(vs.v1)
}
func (vs DualSortFF) Less(i, j int) bool {
	return vs.v1[i] < vs.v1[j] || (fpeq(vs.v1[i], vs.v1[j]) && vs.v2[i] < vs.v2[j])
}
func (vs DualSortFF) Swap(i, j int) {
	vs.v1[i], vs.v1[j] = vs.v1[j], vs.v1[i]
	vs.v2[i], vs.v2[j] = vs.v2[j], vs.v2[i]
}

type DualSortFI struct {
	v1 []float64
	v2 []int
}

func (vs DualSortFI) Len() int {
	return len(vs.v1)
}
func (vs DualSortFI) Less(i, j int) bool {
	return vs.v1[i] < vs.v1[j] || (fpeq(vs.v1[i], vs.v1[j]) && vs.v2[i] < vs.v2[j])
}
func (vs DualSortFI) Swap(i, j int) {
	vs.v1[i], vs.v1[j] = vs.v1[j], vs.v1[i]
	vs.v2[i], vs.v2[j] = vs.v2[j], vs.v2[i]
}

// Returns if a and b are 'equal' for the floating point definition
func fpeq(a float64, b float64) bool {
	// TODO(padster): Move to common utility library.
	rtol, atol := 1e-5, 1e-8
	return math.Abs(a-b) < atol+rtol*math.Abs(b)
}
