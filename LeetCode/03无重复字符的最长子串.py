"""
给定一个字符串 s ，请你找出其中不含有重复字符的 最长 子串 的长度。

示例 1:
输入: s = "abcabcbb"
输出: 3
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

示例 2:
输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。

示例 3:
输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
"""

"""
解题思路：
1.定义两个指针left和right。分别表示当前窗口(子串)的左右边界。初始化时，左右边界都为0
2.使用一个哈希集合set来维护当前窗口内的字符，来判断字符是否重复
3.向右滑动窗口扩展
    如果当前字符不在集合中，说明无重复，右指针右移，扩大窗口，将字符加入集合
    如果当前字符已经存在集合中，说明出现了重复字符，需要收缩窗口，移动左指针，直到移除重复字符
4.记录每次窗口的大小，right-left，保留最大值
5.重复操作直到右指针扫描完整个字符串
"""

def length_of_longest_substring(s: str) -> int:
    char_set = set()                           #存储滑动窗口中的字符
    left = 0                                   #滑动窗口的左边界
    max_length = 0                             #最长子串的长度

    for right in range(len(s)):                #右指针从0开始移动到字符串末尾
        while s[right] in char_set:            #如当前字符重复出现，则移动左指针，直到无重复
            char_set.remove(s[left])           #移除窗口左侧的字符
            left += 1                          #左指针右移

        char_set.add(s[right])                 #将当前字符加入窗口
        max_length = max(max_length, right - left + 1)        #更新最大长度

    return max_length


# 示例 1
s = "abcabcbb"
print(length_of_longest_substring(s))  # 输出: 3

# 示例 2
s = "bbbbb"
print(length_of_longest_substring(s))  # 输出: 1

# 示例 3
s = "pwwkew"
print(length_of_longest_substring(s))  # 输出: 3

# 示例 4: 边界情况
s = ""
print(length_of_longest_substring(s))  # 输出: 0

# 示例 5
s = " "
print(length_of_longest_substring(s))  # 输出: 1

# 示例 6
s = "au"
print(length_of_longest_substring(s))  # 输出: 2

