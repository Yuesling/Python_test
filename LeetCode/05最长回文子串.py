"""
给你一个字符串 s，找到 s 中最长的 回文 子串。

示例 1：
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。

示例 2：
输入：s = "cbbd"
输出："bb"
"""

"""
解题思路：
    采用中心扩展法：
    从中心点开始，同时向左和向右扩展
    扩展的条件是两侧的字符相等且未超出字符串边界
    停止条件是左右字符不等或者超出边界
  中心点确定：
    对于一个长度为n的字符串s，有两类中心
    1.n为奇数：中心为每一个字符，长度为n
    2.n为偶数，中心为两个相邻字符，长度为n-1
    总长度为2n-1
"""

def longestPalindrome(s: str) -> str:
    #辅助函数，从中心扩展，返回回文的起始和结束下标
    def expandAroundCenter(s, left, right):
        #向两边扩展，知道不满足回文条件
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        #返回回文起始和结束下标
        return left + 1, right - 1        #left + 1 是扩展停止时的实际起点

    start, end = 0, 0         #表示最长回文的起始和结束位置
    # 遍历所有扩展中心
    for i in range(len(s)):
        # 1.奇数长度回文，单字符中心
        l1, r1 = expandAroundCenter(s, i, i)
        # 2.偶数长度回文，双字符中心
        l2, r2 = expandAroundCenter(s, i, i + 1)

        # 更新结果，比较两种情况下的回文串长度
        if r1 - l1 > end - start:
            start, end = l1, r1

        if r2 - l2 > end - start:
            start, end = l2, r2
    # 根据起始值和结束值返回最长回文子串
    return s[start : end + 1]

print(longestPalindrome("babad"))
print(longestPalindrome("cbbd"))