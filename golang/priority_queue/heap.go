package priority_queue

import "errors"

type Heap struct {
	data []int
}

// 返回堆大小
func (heap Heap) Size() int {
	return len(heap.data)
}

//  返回是否为空
func (heap Heap) IsEmpty() bool {
	return heap.Size() == 0
}

// 返回parent所在的索引位置
func (heap Heap) Parent(index int) int {
	if index == 0 {
		panic("error")
	}
	return (index - 1) / 2
}

// 返回left child所在的索引位置
func (heap Heap) LeftChild(index int) int {
	return index*2 + 1
}

// 返回right child所在的索引位置
func (heap Heap) RightChild(index int) int {
	return (index + 1) * 2
}

// 添加元素
func (heap *Heap) Add(e int) {
	heap.data = append(heap.data, e)
	heap.shiftUp(len(heap.data) - 1)
}

// 上浮操作
func (heap *Heap) shiftUp(index int)  {
	for index > 0 && heap.data[heap.Parent(index)] < heap.data[index] {
		heap.data[index], heap.data[heap.Parent(index)] = heap.data[heap.Parent(index)], heap.data[index]
		index = heap.Parent(index)
	}
}

// 获取最大元素
func (heap *Heap) ExtractMax()(maxEle int, err error){
	if len(heap.data) == 0{
		return 0, errors.New("empty data")
	}
	maxEle = heap.data[0]
	heap.data[0] = heap.data[len(heap.data) - 1]
	// pop
	heap.data = heap.data[:len(heap.data) - 1]
	heap.shiftDown(0)
	return
}

// 下潜操作
func (heap *Heap) shiftDown(index int){
	for heap.LeftChild(index) < len(heap.data){
		j := heap.LeftChild(index)
		if j + 1 < len(heap.data) && heap.data[j+ 1] > heap.data[j]{
			j = heap.RightChild(index)
		}
		if heap.data[index] >= heap.data[j] {
			break
		}
		heap.data[index], heap.data[j] = heap.data[j], heap.data[index]
		index = j
	}
}




