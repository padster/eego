package util

import (
	"math"
)

// Multiple different data types of (T1, T2) dual arrays, which can be sorted by the 
// first element, with tie-breaker by the second.
// UUUGGH, go missing templates/generics.

// DualSortII allows to sort (int, int) pairs.
type DualSortII struct {
	V1 []int
	V2 []int
}
func (vs DualSortII) Len() int {
	return len(vs.V1)
}
func (vs DualSortII) Less(i, j int) bool {
	return vs.V1[i] < vs.V1[j] || ((vs.V1[i] == vs.V1[j]) && (vs.V2[i] < vs.V2[j]))
}
func (vs DualSortII) Swap(i, j int) {
	vs.V1[i], vs.V1[j] = vs.V1[j], vs.V1[i]
	vs.V2[i], vs.V2[j] = vs.V2[j], vs.V2[i]
}

// DualSortFF allows to sort (float, float) pairs.
type DualSortFF struct {
	V1 []float64
	V2 []float64
}
func (vs DualSortFF) Len() int {
	return len(vs.V1)
}
func (vs DualSortFF) Less(i, j int) bool {
	return vs.V1[i] < vs.V1[j] || (Fpeq(vs.V1[i], vs.V1[j]) && vs.V2[i] < vs.V2[j])
}
func (vs DualSortFF) Swap(i, j int) {
	vs.V1[i], vs.V1[j] = vs.V1[j], vs.V1[i]
	vs.V2[i], vs.V2[j] = vs.V2[j], vs.V2[i]
}

// DualSortFI allows you to sort (float, int) pairs.
type DualSortFI struct {
	V1 []float64
	V2 []int
}
func (vs DualSortFI) Len() int {
	return len(vs.V1)
}
func (vs DualSortFI) Less(i, j int) bool {
	return vs.V1[i] < vs.V1[j] || (Fpeq(vs.V1[i], vs.V1[j]) && vs.V2[i] < vs.V2[j])
}
func (vs DualSortFI) Swap(i, j int) {
	vs.V1[i], vs.V1[j] = vs.V1[j], vs.V1[i]
	vs.V2[i], vs.V2[j] = vs.V2[j], vs.V2[i]
}


// Returns if a and b are 'equal' for the floating point definition
func Fpeq(a float64, b float64) bool {
	// TODO(padster): Move outside dualsort.go
	rtol, atol := 1e-5, 1e-8
	return math.Abs(a-b) < atol+rtol*math.Abs(b)
}
