package priority_queue

import (
	"container/heap"
	"fmt"
	"testing"
)

func TestHeap(t *testing.T) {
	_heap := Heap{data: []int{}}
	for _, data := range ([]int{1, 3, 2, 5}) {
		_heap.Add(data)
	}
	for i := 0; i < 4; i++ {
		fmt.Println(_heap.ExtractMax())
	}
}

// interface...
type Person interface {
	Eat(food string)
	Get() string
}

type XiaoMing struct {
	food string
}

func (xiaoming XiaoMing) Get() string{
	return xiaoming.food
}

func (xiaoming *XiaoMing)Eat(food string){
	xiaoming.food = food
}

func TestPerson(t *testing.T){
	var person Person
	 person = new(XiaoMing)
	 person.Get()
}

// 测试内置的堆
type IntHeap []int

func (intHeap *IntHeap) Len() int {
	return len(*intHeap)
}

// 修改此方法实现最大堆和最小堆
func (intHeap *IntHeap) Less(i, j int) bool {
	return (*intHeap)[i] > (*intHeap)[j]
}

func (intHeap *IntHeap) Swap(i, j int) {
	(*intHeap)[i], (*intHeap)[j] = (*intHeap)[j], (*intHeap)[i]
}

func (intHeap *IntHeap) Push(x interface{}){
	*intHeap = append(*intHeap, x.(int))
}

func (intHeap *IntHeap) Pop() interface{}{
	old := *intHeap
	n := len(old)
	x := old[n - 1]
	*intHeap = old[0: n - 1]
	return x
}

// 测试使用内置的heap来实现最大堆和最小堆
func TestInnerHeap(t *testing.T) {
	h := &IntHeap{2, 1, 5}
	heap.Init(h)
	heap.Push(h, 3)
	fmt.Printf("minimum: %d\n", (*h)[0])
	for h.Len() > 0 {
		fmt.Printf("%d ", heap.Pop(h))
	}
}




func TestMap(t *testing.T){
	type Payload map[string]string
	payload := Payload{"name": "aa", "age": "bb"}
	for key, value := range payload{
		fmt.Println(key, value)
	}
}
