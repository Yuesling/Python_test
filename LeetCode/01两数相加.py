"""
给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
请你将两个数相加，并以相同形式返回一个表示和的链表。
你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.
示例 2：

输入：l1 = [0], l2 = [0]
输出：[0]
示例 3：

输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
输出：[8,9,9,9,0,0,0,1]

"""



# 列表形式可以这么做
# def addTwoNumbers(l1:list, l2:list)->list:
#     num1 = int(''.join(map(str, l1[::-1])))
#     num2 = int(''.join(map(str, l2[::-1])))
#
#     result = num1 + num2
#     #return list(str(result))[::-1]
#     return [int(x) for x in str(result)][::-1]
#
# result1 = addTwoNumbers(l1 = [2,4,3], l2 = [5,6,4])
# print(result1)
# result2 = addTwoNumbers(l1 = [0], l2 = [0])
# print(result2)
# result3 = addTwoNumbers(l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9])
# print(result3)


#如下是链表

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def addTwoNumbers(self, l1 , l2):
        dummy_head = ListNode(0)     #哨兵节点
        current = dummy_head
        carry = 0                     #初始化进位为0

        while l1 or l2:               #遍历l1和l2，直至为空
            x = l1.val if l1 else 0    #如果某个链表已经为空，则补0
            y = l2.val if l2 else 0

            sum = x + y + carry       #计算当前的和，以及进位
            carry = sum // 10         #更新进位
            current.next = ListNode(sum % 10)     #创建当前位的节点
            current = current.next

            if l1:                      #移动到下一个节点
                l1 = l1.next
            if l2:
                l2 = l2.next

        if carry > 0:                   #如果最后还有进位，创建一个额外的节点
            current.next = ListNode(carry)

        return dummy_head.next          #返回结果链表的头节点

l1 = ListNode(2, ListNode(4, ListNode(3)))
l2 = ListNode(5, ListNode(6, ListNode(4)))

result = l1.addTwoNumbers(l2, l1)
while result:
    print(result.val)
    result = result.next


