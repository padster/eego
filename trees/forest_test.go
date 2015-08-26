package trees

import (
	"testing"
)


func TestSplit(t *testing.T) {
	f := NewForest(2, 1, 0)
	f.Train([]int{
		10, 15, 11, 12, 8, 3, 7,
	}, []int{
		 0,  1,  0,  1, 0, 0, 1,
	})
	t.Error("Test run")
}
