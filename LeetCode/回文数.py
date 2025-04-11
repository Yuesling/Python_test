"""
给你一个整数 x ，如果 x 是一个回文整数，返回 true ；否则，返回 false 。

回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。

例如，121 是回文，而 123 不是。
"""

"""
解题思路：
1.直接采用整数反转的方式
    长度是偶数，则反转后半部分，来判断前后半部分是否相等
    长度是奇数，以中间为基准，反转后半部分来判断
"""

def is_palindrome(x: int) -> bool:
    if x < 0 or (x % 10 == 0 and x != 0):     #负数 或以0结尾的非回文数
        return False

    reversed_half = 0
    while x > reversed_half:                   #遍历数字直到数字的左半部分小于等于右半部分
        reversed_half = reversed_half *10 + x % 10           #反转右半部分
        x = x // 10                                          #去掉原数字的最后一位

    return x == reversed_half or x == reversed_half // 10    #比较原始数字的左半部分和右半部分


print(is_palindrome(12321))
print(is_palindrome(-1))
print(is_palindrome(123))
print(is_palindrome(12210))