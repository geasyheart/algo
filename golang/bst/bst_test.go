package main

import (
	"fmt"
	"testing"
)

var (
	dataExample = []int{1, 3, 2, 4, 5}
)

func TestBinarySearchTree_Add(t *testing.T) {
	var binarySearchTree BinarySearchTree
	for _, data := range dataExample{
		binarySearchTree.Add(data)
	}
}

func TestBinarySearchTree_Contains(t *testing.T) {
	var binarySearchTree BinarySearchTree
	for _, data := range dataExample{
		binarySearchTree.Add(data)
	}
	for _, data := range dataExample{
		fmt.Println(binarySearchTree.Contains(data) == true)
	}
	fmt.Println(binarySearchTree.Contains(123) == false)
}

func TestBinarySearchTree_ContainsUsePointer(t *testing.T) {
	var binarySearchTree BinarySearchTree
	for _, data := range dataExample{
		binarySearchTree.Add(data)
	}
	for _, data := range dataExample{
		fmt.Println(binarySearchTree.ContainsUsePointer(data) == true)
	}
	fmt.Println(binarySearchTree.ContainsUsePointer(123) == false)
}

func TestBinarySearchTree_PreOrder(t *testing.T) {
	var binarySearchTree BinarySearchTree
	for _, data := range dataExample{
		binarySearchTree.Add(data)
	}
	fmt.Println("----------preOrder-------")
	binarySearchTree.PreOrder()
	fmt.Println("-----------------")
}

func TestBinarySearchTree_LevelOrder(t *testing.T) {
	var binarySearchTree BinarySearchTree
	for _, data := range dataExample{
		binarySearchTree.Add(data)
	}
	binarySearchTree.LevelOrder()

}

func TestBinarySearchTree_Maximum(t *testing.T) {
	var binarySearchTree BinarySearchTree
	for _, data := range dataExample{
		binarySearchTree.Add(data)
	}
	fmt.Println("max value: ", binarySearchTree.Maximum().E == 5)
	fmt.Println("min value: ", binarySearchTree.Minimum().E == 1)

}

func TestBinarySearchTree_DelMin(t *testing.T) {
	var binarySearchTree BinarySearchTree
	for _, data := range dataExample{
		binarySearchTree.Add(data)
	}
	binarySearchTree.DelMin()
	binarySearchTree.LevelOrder()


	binarySearchTree.Root = nil
	for _, data := range []int{3, 2, 1, 4, 5}{
		binarySearchTree.Add(data)
	}
	binarySearchTree.DelMin()
	binarySearchTree.LevelOrder()
}


func TestBinarySearchTree_DelMax(t *testing.T) {
	var binarySearchTree BinarySearchTree
	for _, data := range []int{3, 2, 1, 4, 5}{
		binarySearchTree.Add(data)
	}
	binarySearchTree.DelMax()
	binarySearchTree.LevelOrder()
}
