package main

import (
	"fmt"
	"log"
)

type Node struct {
	E     int
	Left  *Node
	Right *Node
}

func (node Node) String() string {
	return fmt.Sprintf("<Node> %d", node.E)
}

type BinarySearchTree struct {
	Root *Node
}

// 增加node
func (binarySearchTree *BinarySearchTree) Add(e int) {
	node := Node{e, nil, nil}
	if binarySearchTree.Root == nil {
		binarySearchTree.Root = &node
	} else {
		add(binarySearchTree.Root, e)
	}

}

func add(node *Node, e int) {
	if node.E < e && node.Right == nil {
		node.Right = &Node{e, nil, nil}
	} else if node.E < e && node.Right != nil {
		add(node.Right, e)
	} else if node.E > e && node.Left == nil {
		node.Left = &Node{e, nil, nil}
	} else if node.E > e && node.Left != nil {
		add(node.Left, e)
	} else {
		// equal
	}
}

// contain
func (binarySearchTree BinarySearchTree) Contains(e int) bool {
	return contains(*binarySearchTree.Root, e)
}

func contains(node Node, e int) bool {
	if node.E == e {
		return true
	} else if node.E < e && node.Right != nil {
		return contains(*node.Right, e)
	} else if node.E > e && node.Left != nil {
		return contains(*node.Left, e)
	} else if node.E < e && node.Right == nil {
		return false
	} else if node.E > e && node.Left == nil {
		return false
	} else {
		return false
	}
}

// 使用指针的方式
func (binarySearchTree *BinarySearchTree) ContainsUsePointer(e int) bool {
	return containsUsePointer(binarySearchTree.Root, e)
}

func containsUsePointer(node *Node, e int) bool {
	if node == nil {
		return false
	}
	if node.E == e {
		return true
	} else if node.E > e {
		return containsUsePointer(node.Left, e)
	} else {
		return containsUsePointer(node.Right, e)
	}
}

// 前序遍历
func (binarySearchTree *BinarySearchTree) PreOrder() {
	preOrder(binarySearchTree.Root)
}

// 中序遍历
func (binarySearchTree *BinarySearchTree) InOrder() {
	inOrder(binarySearchTree.Root)
}

func preOrder(node *Node) {
	if node == nil {
		return
	}
	log.Println(node.E)
	preOrder(node.Left)
	preOrder(node.Right)
}

func inOrder(node *Node) {
	if node == nil {
		return
	}
	inOrder(node.Left)
	log.Println(node.E)
	inOrder(node.Right)
}

func (binarySearchTree *BinarySearchTree) PostOrder() {
	postOrder(binarySearchTree.Root)
}

func postOrder(node *Node) {
	if node == nil {
		return
	}
	postOrder(node.Left)
	postOrder(node.Right)
	log.Println(node.E)
}

func (binarySearchTree *BinarySearchTree) LevelOrder() {
	firstLevel := []*Node{binarySearchTree.Root}
	levelOrder(firstLevel)
}

func levelOrder(nodes []*Node) {
	if len(nodes) == 0 {
		return
	}
	// for print
	var level []*Node
	var nextLevel []*Node
	for _, nodeP := range nodes {
		level = append(level, nodeP)
		if nodeP.Left != nil {
			nextLevel = append(nextLevel, nodeP.Left)
		}
		if nodeP.Right != nil {
			nextLevel = append(nextLevel, nodeP.Right)
		}
	}
	log.Println("current level: ", level)
	levelOrder(nextLevel)
}

func (binarySearchTree *BinarySearchTree) Minimum() *Node {
	return minimum(binarySearchTree.Root)
}

func minimum(node *Node) *Node {
	if node.Left != nil {
		return minimum(node.Left)
	} else {
		return node
	}
}

func (binarySearchTree *BinarySearchTree) Maximum() *Node {
	return maximum(binarySearchTree.Root)
}

func maximum(node *Node) *Node {
	if node.Right != nil {
		return maximum(node.Right)
	} else {
		return node
	}
}

func (binarySearchTree *BinarySearchTree) DelMin() {
	binarySearchTree.Root = delMin(binarySearchTree.Root)
}

func delMin(node *Node) *Node {
	if node.Left == nil {
		right := node.Right
		node.Right = nil
		return right
	}
	node.Left = delMin(node.Left)
	return node
}

func (binarySearchTree *BinarySearchTree) DelMax() {
	binarySearchTree.Root = delMax(binarySearchTree.Root)
}

func delMax(node *Node) *Node{
	if node.Right == nil {
		left:= node.Left
		node.Left = nil
		return left
	}
	node.Right = delMax(node.Right)
	return node
}


func (binarySearchTree *BinarySearchTree) DelNode(node *Node) {
	return
}
