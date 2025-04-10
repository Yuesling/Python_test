class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self, ):
        self.head = None

    def printList(self):
        current = self.head
        while current:
            if current.next:
                print(current.val, end="->")
            else:
                print(current.val, end="")
            current = current.next
        print()

    def append(self, val):
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def prepend(self, val):
        new_node = ListNode(val)
        new_node.next = self.head
        self.head = new_node

    def delete(self, val):
        if not self.head:
            return
        if self.head.val == val:
            self.head = self.head.next
            return

        current = self.head
        while current.next and current.next.val == val:
            current = current.next
        if current.next:
            current.next = current.next.next
            return

# 创建节点



l1 = LinkedList()
l1.append(1)
l1.append(2)
l1.append(3)
l1.printList()

l1.prepend(0)
l1.printList()

l1.delete(2)
l1.printList()

