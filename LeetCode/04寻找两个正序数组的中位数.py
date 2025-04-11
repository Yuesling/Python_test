"""
给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
算法的时间复杂度应该为 O(log (m+n)) 。

示例 1：
输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3] ，中位数 2

示例 2：
输入：nums1 = [1,2], nums2 = [3,4]
输出：2.50000
解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
"""

"""
解题思路：
    1.合并数组并排序
    2.如果数组长度为奇数，直接返回中位数
    3.如果数组长度为偶数，
"""

def findMedianSortedArrays(nums1, nums2):
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    low, high = 0, m

    while low <= high:
        midx = (low +high) // 2
        midy = (m + n + 1) // 2 - midx

        maxleftx = float("-inf") if midx == 0 else nums1[midx -1]
        minrightx = float("inf") if midx == m else nums1[midx]

        maxlefty = float("-inf") if midy == 0 else nums2[midy -1]
        minrighty = float("inf") if midy == n else nums2[midy]

        if maxleftx <= minrighty and maxlefty <= minrightx:

            if (m + n) % 2 == 1:
                return max(maxleftx, maxlefty)
            else:
                return (max(maxleftx, maxlefty) + min(minrightx, minrighty)) / 2

        elif maxleftx > minrighty:
            high = midx -1
        else:
            low = midx + 1

    raise ValueError("Input arrays are not sorted.")

nums1 = [1, 3]
nums2 = [2]
print(findMedianSortedArrays(nums1, nums2))  # 输出: 2.0

nums1 = [1, 2]
nums2 = [3, 4]
print(findMedianSortedArrays(nums1, nums2))  # 输出: 2.5

nums1 = [0, 0]
nums2 = [0, 0]
print(findMedianSortedArrays(nums1, nums2))  # 输出: 0.0

nums1 = []
nums2 = [1]
print(findMedianSortedArrays(nums1, nums2))  # 输出: 1.0

nums1 = [2]
nums2 = []
print(findMedianSortedArrays(nums1, nums2))  # 输出: 2.0



