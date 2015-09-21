package trees

import (
	"container/heap"
	"fmt"
	"sort"

	"github.com/padster/eego/util"
)

/*
Custom implementation of a Random-forest-ish classifier.

This can either:
 a) Train, by forming a forest off a stream of (float) inputs and (0/1) outputs
 b) Classify, by taking float samples and returning the [0, 1] probability classification for each.

Parameters are:
 - N = frame size.
 - T = tree count.
 - S = max node count for the trees

Training takes place by generating multiple trees off the input, by forming off frames from the input.
An array of size D is created from each frame, by combining:
  - N values in the frame
  - N - 1 differences
  - 1 mean
  - ... other features? auto-detect?

T trees are then created, each given access to look at a subset (~sqrt(D)) of indexes 
in the generated array for each frame. A decision tree is formed by finding the best
index to split off, and doing that continuosly until each has at most S nodes.

Classification happens by, for each input, running the last N frames (zero-padded)
through the trees, and combining the results into an overall prediction.
*/

// Remaining:
//  - Algorithm to pick allowed sets
//  - Create child nodes for leaf -> branch
//  - test!

// TODO - entropy instead of miscalculation? 
// from here: http://www.saedsayad.com/decision_tree.htm

// DOCS
type Forest struct {
	frameSize int
	treeCount int
	minMisclassified int

	leafQueue nodeQueue
	allowed [][]int

	roots nodeQueue

	// current training state
	trainFrameCount int
	trainSamples []int
	trainExpected []int
}

// DOCS - Node of a tree within the forest.
type node struct {
	// Parent node for this decision node, nil for tree root
	parent *node
	// List of frames that made it here.
	inputs []int
	// Classify as 1 (true) or 0 (false)
	classifyAsTrue bool
	// How many are misclassified at this point in the tree
	misclassified int
	// Data specific to branches
	branchData branchNode
	// Whether it's a leaf or branch node.
	isLeaf bool
	// Which tree this comes from
	originalRoot int
}

// DOCS
type branchNode struct {
	// Index to decide on
	decideFeature int
	// Value to switch on, < decideCutoff go to lowerChild.
	decideCutoff int

	// Next decision to make if this decision passes (branches)
	lowerChild *node
	// Next decision to make if this decision fails (branches)
	highEqChild *node
}

// DOCS
func NewForest(frameSize int, treeCount int, minMisclassified int) *Forest {
	features := 2 * frameSize - 1
	allowed := make([][]int, treeCount, treeCount)

	// TODO - generate forbidden lists
	if treeCount != 1 {
		panic("Forest currently only supports single tree")
	}
	allowed[0] = make([]int, features, features)
	for i := 0; i < features; i++ {
		allowed[0][i] = i
	}

	f := Forest{
		frameSize,
		treeCount,
		minMisclassified,
		make(nodeQueue, treeCount),
		allowed,
		make(nodeQueue, treeCount),
		// These get filled in when training starts:
		-1,
		nil,
		nil,
	}
	return &f
}

// DOCS
func (f *Forest) Train(samples []int, expected []int) {
	// Train-scoped variables:
	f.trainSamples  = samples
	f.trainExpected = expected
	f.trainFrameCount = len(samples) - f.frameSize + 1

	// Initial state for root nodes of each tree:
	trueCount := 0
	for i := 0; i < f.trainFrameCount; i++ {
		if expected[i + f.frameSize - 1] == 1 {
			trueCount++
		}
	}
	moreTrue := trueCount > (f.trainFrameCount - trueCount)
	misclassified := trueCount
	if moreTrue {
		misclassified = f.trainFrameCount - trueCount
	}
	// fmt.Printf("moreTrue = %v, misclassified = %v\n", moreTrue, misclassified)

	// Create each root node separately:
	for i := 0; i < f.treeCount; i++ {
		// fmt.Printf("Creating node %d\n", i)
		f.roots[i] = &node{
			nil,
			make([]int, f.trainFrameCount, f.trainFrameCount),
			moreTrue, // classifyAsTrue
			misclassified,
			branchNode{
				-1, -1,
				nil, nil,
			},
			true, // isLeaf
			i, // originalRoot
		}
		f.leafQueue[i] = f.roots[i]

		// Pre-fill inputs and initial best split point.
		for j := 0; j < f.trainFrameCount; j++ {
			f.leafQueue[i].inputs[j] = j
		}
		f.leafQueue[i].precalcBestSplit(f)
	}

	// Split the nodes until we're close enough:
	// fmt.Printf("Initting heap...\n")
	heap.Init(&f.leafQueue)
	for len(f.leafQueue) > 0 {
		nextLeaf := heap.Pop(&f.leafQueue).(*node)
		// fmt.Printf("Splitting node which misclassifies %d\n", nextLeaf.misclassified)
		if nextLeaf.branchData.decideFeature == -1 {
			// Nothing left to split, we've done as much as possible.
			break
		}
		if nextLeaf.misclassified < f.minMisclassified {
			// Only rounding error left
			break
		}
		nextLeaf.convertToBranch(f)
	}
}

// DOCS - Number of nodes in the entire forest
func (f *Forest) DecisionNodes() int {
	count := 0
	for _, n := range f.roots {
		count += n.subtreeSize()
	}
	return count
}

// DOCS - average miscalculations across all roots
func (f *Forest) AverageErrors() float64 {
	errors := 0
	for _, n := range f.roots {
		errors += n.totalErrors()
	}
	return float64(errors) / float64(len(f.roots))
}

// DOCS - fill in the branch node data with the best split decision
func (n *node) precalcBestSplit(f *Forest) {
	// fmt.Printf("!!!Presplitting node %v\n", n)
	// Find all remaining features that we can decide on:
	allowed := map[int]bool{}
	for _, v := range f.allowed[n.originalRoot] {
		allowed[v] = true
	}
	// PICK: remove allowed nodes?
	// for at := n.parent; at != nil; at = at.parent {
		// fmt.Printf("Used %d at parent %v\n", at.branchData.decideFeature, at)
		// delete(allowed, at.branchData.decideFeature)
	// }

	// Nothing left to split on, so don't
	if len(allowed) == 0 {
		// fmt.Printf("No features left!\n")
		return
	}

	// fmt.Printf("Allowed = {")
	// for f := range allowed {
		// fmt.Printf("%d, ", f)
	// }
	// fmt.Printf("}\n")

	// Find the best of those, which is also a big enough improvement.
	upperBar := int(float64(n.misclassified) * 0.99) // need to at least be fix 1%

	bestSplit := splitDetails{-1, -1, false, upperBar, -1, -1}
	for splitFeature := range allowed {
		nextSplit := n.splitReduction(f, splitFeature)
		if nextSplit.misses < bestSplit.misses {
			bestSplit = nextSplit
		}
	}

	// Split, but only if it improves things:
	if bestSplit.splitFeature != -1 {
		// fmt.Printf("Performing presplit! On feature %d\n", bestSplit.splitFeature)
		n.presplitOn(f, bestSplit)
	}
}

// HACK
type splitDetails struct {
	splitValue int
	splitFeature int
	trueBelow bool
	misses int
	missesBelow int
	missesAbove int
}

// DOCS - misclassified improvement given a feature to split
func (n *node) splitReduction(f *Forest, feature int) splitDetails {
	// fmt.Printf("Trying to split %v on feature %d\n", n, feature)
	nFrames := len(n.inputs)

	// Sort, find best split, then return new misclassification details.
	trueBelow, trueAbove := 0, n.misclassified
	falseBelow, falseAbove := 0, nFrames - n.misclassified
	if n.classifyAsTrue {
		trueAbove = nFrames - n.misclassified
		falseAbove = n.misclassified
	}
	// fmt.Printf("TB/TA/FB/FA = %d/%d/%d/%d\n", 
		// trueBelow, trueAbove, falseBelow, falseAbove)

	// currentWrong := n.misclassified
	dsii := util.DualSortII {
		make([]int, nFrames, nFrames),
		make([]int, nFrames, nFrames),
	}

	// Find the value for each frame for the given feature:
	for i, frame := range n.inputs {
		dsii.V1[i] = scoreForFrameAndFeature(f, frame, feature)
		dsii.V2[i] = frame
	}
	sort.Sort(dsii)
	// fmt.Printf("scores = %v\n", dsii.V1)
	// fmt.Printf("indexs = %v\n", dsii.V2)

	// HACK - remove
	tmp := make([]int, nFrames, nFrames)
	for i := 0; i < nFrames; i++ {
		tmp[i] = f.trainExpected[dsii.V2[i] + f.frameSize - 1]
	}
	// fmt.Printf("output = %v\n", tmp)


	bestSplit := splitDetails{-1, -1, false, n.misclassified, -1, -1}

	for splitBefore := 0; splitBefore < nFrames; splitBefore++ {
		// Splitting on the same value isn't allowed, numbers are wrong.
		considerSplit := true
		thisSplit := dsii.V1[splitBefore]
		if splitBefore > 0 {
			lastSplit := dsii.V1[splitBefore - 1]
			if thisSplit == lastSplit {
				// fmt.Printf("Skipping %d\n", thisSplit)
				considerSplit = true
			}
		}

		// Derive miscalculations based on splitting here
		if considerSplit {
			missAsFalseBelow := trueBelow + falseAbove
			missAsTrueBelow := falseBelow + trueAbove
			// fmt.Printf("Trying split at %d, missTB, missFB = %d, %d\n", 
				// thisSplit, missAsTrueBelow, missAsFalseBelow)
			if missAsTrueBelow < missAsFalseBelow {
				if missAsTrueBelow < bestSplit.misses {
					bestSplit = splitDetails{
						thisSplit, feature, true, 
						missAsTrueBelow, falseBelow, trueAbove,
					}
				}
			} else {
				if missAsFalseBelow < bestSplit.misses {
					bestSplit = splitDetails{
						thisSplit, feature, false, 
						missAsFalseBelow, trueBelow, falseAbove,
					}
				}
			}
		}

		frame := dsii.V2[splitBefore] + f.frameSize - 1
		if f.trainExpected[frame] == 1 {
			trueBelow++
			trueAbove--
		} else {
			falseBelow++
			falseAbove--
		}
	}

	// fmt.Printf("Best split found: f[%d] < %d, classifying below as %v\n", 
		// bestSplit.splitFeature, bestSplit.splitValue, bestSplit.trueBelow)
	return bestSplit
}

// DOCS - split a node on a given feature
func (n *node) presplitOn(f *Forest, split splitDetails) {
	fmt.Printf("Splitting node with %d mis, by: %v\n", n.misclassified, split)

	lo, hi := 0, len(n.inputs) - 1
	for lo < hi {
		for ; lo < hi; lo++ {
			score := scoreForFrameAndFeature(f, n.inputs[lo], split.splitFeature)
			isBelow := score < split.splitValue
			// In the wrong place if isBelow == true && trueBelow == false, or
			// isBelow == false && trueBelow == true
			if isBelow != split.trueBelow {
				break
			}
		}
		for ; lo < hi; hi-- {
			score := scoreForFrameAndFeature(f, n.inputs[hi], split.splitFeature)
			isBelow := score < split.splitValue
			if isBelow == split.trueBelow {
				break
			}
		}
		if lo != hi {
			// fmt.Printf("Swapping in[%d]=%d with in[%d]=%d\n", 
				// lo, n.inputs[lo], hi, n.inputs[hi])
			n.inputs[lo], n.inputs[hi] = n.inputs[hi], n.inputs[lo]
		}
	}
	for ; lo < len(n.inputs); lo++ {
		score := scoreForFrameAndFeature(f, n.inputs[lo], split.splitFeature)
		isBelow := score < split.splitValue
		if isBelow != split.trueBelow {
			break
		}
		// fmt.Printf("Bumping slice point to %d\n", lo)
	}

	slicePoint := lo
	// fmt.Printf("Slice point found! %d\n", slicePoint)

	n.branchData.decideFeature = split.splitFeature
	n.branchData.decideCutoff = split.splitValue
	n.branchData.lowerChild = &node{
		n,
		n.inputs[:slicePoint],
		split.trueBelow,
		split.missesBelow,
		branchNode{-1, -1, nil, nil},
		true, // isLeaf,
		n.originalRoot,
	}
	n.branchData.highEqChild = &node{
		n,
		n.inputs[slicePoint:],
		!split.trueBelow,
		split.missesAbove,
		branchNode{-1, -1, nil, nil},
		true, // isLeaf,
		n.originalRoot,
	}
	// fmt.Printf("Created two children:\n\t<\t%v\n\t>=\t%v\n", n.branchData.lowerChild, n.branchData.highEqChild)
}

// DOCS - number of misclassified frames
func (n *node) totalErrors() int {
	if n.isLeaf {
		return n.misclassified
	} else {
		return n.branchData.lowerChild.totalErrors() + n.branchData.highEqChild.totalErrors()
	}
}

func (n *node) subtreeSize() int {
	count := 1
	if !n.isLeaf {
		count += n.branchData.lowerChild.subtreeSize()
		count += n.branchData.highEqChild.subtreeSize()
	}
	return count
}

// DOCS - pull out a feature for a given frame
func scoreForFrameAndFeature(f *Forest, frame int, feature int) int {
	// PICK - apply another mapping, i.e. use frame + MAP[feature] not frame + feature?
	if feature < f.frameSize {
		return f.trainSamples[frame + feature]
	} else if (feature - f.frameSize) < (f.frameSize - 1) {
		first := frame + (feature - f.frameSize)
		return f.trainSamples[first + 1] - f.trainSamples[first]
	} else {
		panic("TODO - support more features?")
	}
}


// DOCS - this leaf node is being converted into a decision one instead.
func (n *node) convertToBranch(f *Forest) {
	// TODO - don't convert if it makes things worse.
	n.isLeaf = false
	// fmt.Printf("Converting to branch, pre-calc split both children\n")
	lowerChild, upperChild := n.branchData.lowerChild, n.branchData.highEqChild
	if lowerChild.misclassified > 0 {
		lowerChild.precalcBestSplit(f)
		if lowerChild.branchData.decideFeature != -1 {
			heap.Push(&f.leafQueue, lowerChild)
		}
	}
	if upperChild.misclassified > 0 {
		upperChild.precalcBestSplit(f)
		if upperChild.branchData.decideFeature != -1 {
			heap.Push(&f.leafQueue, upperChild)
		}	
	}
}

// Priority queue for leaf nodes:
type nodeQueue []*node

func (pq *nodeQueue) IsEmpty() bool {
    return len(*pq) == 0
}

func (pq *nodeQueue) Push(i interface{}) {
	item := i.(*node)
	*pq = append(*pq, item)
}

func (pq *nodeQueue) Pop() interface{} {
	n := len(*pq)
    r := (*pq)[n - 1]
    *pq = (*pq)[0 : n - 1]
    return r
}

func (pq *nodeQueue) Len() int {
    return len(*pq)
}

// post: true iff is i less than j
func (pq *nodeQueue) Less(i, j int) bool {
    I := (*pq)[i]
    J := (*pq)[j]
    iFix := I.misclassified - (
    	I.branchData.lowerChild.misclassified +
    	I.branchData.highEqChild.misclassified)
    jFix := J.misclassified - (
    	J.branchData.lowerChild.misclassified +
    	J.branchData.highEqChild.misclassified)
    return iFix > jFix
}

func (pq *nodeQueue) Swap(i, j int) {
    (*pq)[i], (*pq)[j] = (*pq)[j], (*pq)[i]
}

/* HACK
func (pq *nodeQueue) String() string {
    var build string = "{"
    for _, v := range *pq {
        build += v.inputs
    }
    build += "}"
    return build
}
*/
