package priority_queue

import (
	"container/heap"
	"fmt"
	"testing"
)

// 使用内置的heap来实现优先队列

// An Item is something we manage in a priority queue.
type Item struct {
	value    map[string]interface{}
	priority int // The priority of the item in the queue.
	// The index is needed by update and is maintained by the heap.Interface methods.
	index int // The index of the item in the heap.
}

// A PriorityQueue implements heap.Interface and holds Items.
type PriorityQueue []*Item

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	// We want Pop to give us the highest, not lowest, priority so we use greater than here.
	return pq[i].priority > pq[j].priority
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *PriorityQueue) Push(x interface{}) {
	n := len(*pq)
	item := x.(*Item)
	item.index = n
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	item.index = -1 // for safety
	*pq = old[0 : n-1]
	return item
}

// update modifies the priority and value of an Item in the queue.
func (pq *PriorityQueue) update(item *Item, value map[string]interface{}, priority int) {
	item.value = value
	item.priority = priority
	heap.Fix(pq, item.index)
}

// This example creates a PriorityQueue with some items, adds and manipulates an item,
// and then removes the items in priority order.
func Test_priorityQueue(t *testing.T) {
	// Some items and their priorities.
	// Example:
	// {
	//    priority: payload
	// }
	//
	//
	items := []map[int]map[string]interface{}{
		{3: {"banana": "3"}},
		{2: {"apple": "2"}},
		{4: {"pear": "4"}},
	}
	pq := make(PriorityQueue, len(items))

	for index, payload := range items {
		for priority, _payload := range payload {
			pq[index] = &Item{
				value:    _payload,
				priority: priority,
				index:    index,
			}
		}
	}
	heap.Init(&pq)

	// Insert a new item and then modify its priority.
	item := &Item{
		value: map[string]interface{}{"orange": "b"},
		priority: 1,
		index:4,
	}
	heap.Push(&pq, item)
	pq.update(item, item.value, 5)

	// Take the items out; they arrive in decreasing priority order.
	for pq.Len() > 0 {
		item := heap.Pop(&pq).(*Item)
		fmt.Printf("%.2d:%s \n", item.priority, item.value)
	}
	// Output:
	// 05:map[orange:b]
	// 04:map[pear:4]
	// 03:map[banana:3]
	// 02:map[apple:2]
}
