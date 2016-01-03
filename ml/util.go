package ml

import (
	"math"
)

func DistSq(v1, v2 []float64) float64 {
	d := 0.0
	for i, _ := range v1 {
		delta := v1[i] - v2[i]
		d += delta * delta
	}
	return d
}

func Dist(v1, v2 []float64) float64 {
	return math.Sqrt(DistSq(v1, v2)) 
}
