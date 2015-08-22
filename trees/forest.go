package trees

import (
	"container/heap"
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

	// current training state
	trainFrameCount int
	trainSamples []int
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
	// Value to switch on, higher go to passChild.
	decideCutoff int

	// Next decision to make if this decision passes (branches)
	passChild *node
	// Next decision to make if this decision fails (branches)
	failChild *node
}

// DOCS
func NewForest(frameSize int, treeCount int, minMisclassified int) *Forest {
	// TODO - generate forbidden lists
	allowed := make([][]int, treeCount, treeCount)

	f := Forest{
		frameSize,
		treeCount,
		minMisclassified,
		make(nodeQueue, treeCount),
		allowed,
		// These get filled in when training starts:
		-1,
		nil,
	}
	return &f
}

// DOCS
func (f *Forest) Train(samples []int, expected []int) {
	// Train-scoped variables:
	f.trainSamples  = samples
	f.trainFrameCount = len(samples) - f.frameSize + 1

	// Initial state for root nodes of each tree:
	trueCount := 0
	for i := 0; i < f.trainFrameCount; i++ {
		if expected[i + f.frameSize] == 1 {
			trueCount++
		}
	}
	moreTrue := trueCount > (f.trainFrameCount - trueCount)
	misclassified := trueCount
	if moreTrue {
		misclassified = f.trainFrameCount - trueCount
	}

	// Create each root node separately:
	for i := 0; i < f.treeCount; i++ {
		f.leafQueue[i] = &node{
			nil,
			make([]int, f.trainFrameCount, f.trainFrameCount),
			moreTrue, // classifyAsTrue
			misclassified,
			branchNode{
				0, 0, 
				nil, nil,
			},
			true, // isLeaf
			i, // originalRoot
		}

		// Pre-fill inputs and initial best split point.
		for j := 0; j < f.trainFrameCount; j++ {
			f.leafQueue[i].inputs[j] = j
		}
		f.leafQueue[i].precalcBestSplit(f)
	}

	// Split the nodes until we're close enough:
	heap.Init(&f.leafQueue)
	for {
		nextLeaf := heap.Pop(&f.leafQueue).(*node)
		if nextLeaf.misclassified < f.minMisclassified {
			break
		}
		nextLeaf.convertToBranch(&f.leafQueue)
	}
}

// DOCS - fill in the branch node data with the best split decision
func (n *node) precalcBestSplit(f *Forest) {
	// Find all remaining features that we can decide on:
	allowed := map[int]bool{}
	for _, v := range f.allowed[n.originalRoot] {
		allowed[v] = true
	}
	for at := n.parent; at != nil; at = at.parent {
		delete(allowed, at.branchData.decideFeature)
	}

	// Find the best of those:
	bestSplitFeature, bestReduction := -1, -1
	for splitFeature := range allowed {
		reduction := n.splitReduction(f, splitFeature)
		if bestSplitFeature == -1 || reduction > bestReduction {
			bestSplitFeature = splitFeature
			bestReduction = reduction
		}
	}

	// Split, but only if it improves things:
	if bestReduction > 0 {
		n.presplitOn(f, bestSplitFeature)
	}
}

// DOCS - misclassified improvement given a feature to split
func (n *node) splitReduction(f *Forest, feature int) int {
	nFrames := f.trainFrameCount
	// currentWrong := n.misclassified
	dsii := util.DualSortII {
		make([]int, nFrames, nFrames),
		make([]int, nFrames, nFrames),
	}

	// Find the value for each frame for the given feature:
	for _, frame := range n.inputs {
		dsii.V1 = append(dsii.V1, scoreForFrameAndFeature(f, frame, feature))
		dsii.V2 = append(dsii.V2, frame)
	}

	// Sort, find best split, then return new misclassification details.
	sort.Sort(dsii)
	// TODO - find best split
	return 0
}

// DOCS - split a node on a given feature
func (n *node) presplitOn(f *Forest, feature int) {
	// TODO - Do weird sorting thing to inputs on n
	slicePoint := 0
	passIsTrue := true

	n.branchData.passChild = &node{
		n,
		n.inputs[:slicePoint],
		passIsTrue,
		0, // TODO - misclassification for pass
		branchNode{},
		true, // isLeaf,
		n.originalRoot,
	}
	n.branchData.failChild = &node{
		n,
		n.inputs[slicePoint:],
		!passIsTrue,
		0, // TODO - misclassification for fail
		branchNode{},
		true, // isLeaf,
		n.originalRoot,
	}

	// TODO - anything else?
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
func (n *node) convertToBranch(leafQueue *nodeQueue) {
	// TODO - don't convert if it makes things worse.
	n.isLeaf = false
	heap.Push(leafQueue, n.branchData.passChild)
	heap.Push(leafQueue, n.branchData.failChild)
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
    	I.branchData.passChild.misclassified +
    	I.branchData.failChild.misclassified)
    jFix := J.misclassified - (
    	J.branchData.passChild.misclassified +
    	J.branchData.failChild.misclassified)
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
