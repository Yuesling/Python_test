"""给定一个整数数组nums和一个整数目标值target，请你在该数组中找出
和为目标值target的那两个整数，并返回它们的数组下标。
你可以假设每种输入只会对应一个答案，并且你不能使用两次相同的元素。

你可以按任意顺序返回答案。

示例1：
输入：nums = [2, 7, 11, 15], target = 9
输出：[0, 1]
解释：因为
nums[0] + nums[1] == 9 ，返回[0, 1] 。

示例2：
输入：nums = [3, 2, 4], target = 6
输出：[1, 2]

示例3：
输入：nums = [3, 3], target = 6
输出：[0, 1]
"""

"""
使用一个字典 hash_map 来存储数组中的值和对应的索引位置。

键（key）：数组中的值
值（value）：数组中的对应索引
遍历数组 nums 时：

计算当前数 num 的补数 complement 值，即 target - num。
检查 complement 是否已经存在于字典 hash_map 中。
如果存在，说明当前数字和字典中的补数加起来就是目标值 target。
直接返回补数的索引和值的索引 [hash_map[complement], index]。
如果不存在，将当前数字存入 hash_map，以备后续使用。
复杂度：

时间复杂度：O(n)，需要单次遍历数组。
空间复杂度：O(n)，在最坏情况下需要存储整个数组的元素到字典中。

"""


def getSum(nums:list, target:int) -> list:

    hash_map = {}                           #空字典用来存放值和索引
    for index, num in enumerate(nums):
        complement = target - num            #计算差值

        if complement in hash_map:           #判断差值是否在哈希表里
            return [hash_map[complement], index]   #如果在，返回差值的索引和当前数字的索引

        hash_map[num] = index                  #如果不存在，将当前数字存在哈希表，键为数字，值为索引

    return []

result1 = getSum([2, 4, 6, 7, 9], 9)
result2 = getSum([3, 2, 4], 6)
result3 = getSum([3, 3], 6)

print(result1)
print(result2)
print(result3)

