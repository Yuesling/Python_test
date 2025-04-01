# 二分法的典型场景是用来在一个有序数组中查找目标值的位置 也叫二分查找
# 二分法将待查找的范围一分为二，判断目标在哪一半范围内，然后逐步缩小查找区间，
# 直到找到目标或者目标不存在

# 前提是数组已经是排好序的 如果不是可以先用sort()方法排序


def binary_search(sorted_list, target):
    """
    二分查找算法实现
    :param sorted_list:排好序的列表
    :param target:要查找的目标值
    :return:如果存在返回目标索引，不存在返回-1
    """
    left, right = 0, len(sorted_list) - 1    #定义左边界和右边界
    while left <= right:                      #当左边界不超过右边界
        mid = left + (right - left) // 2      #计算中间值的索引，直接left+right可能溢出
        mid_value = sorted_list[mid]          #计算中间值
        if mid_value == target:               #找到目标
            return mid
        elif mid_value < target:              #如果中间值小于目标值，目标在右半区
            left = mid + 1
        else:
            right = mid - 1                    #中间值大于目标值，目标在左半区
    return -1

sorted_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
target = 10

result = binary_search(sorted_list, target)
if result != -1:
    print(f'目标值{target}在数组中的索引是{result}')
else:
    print(f'目标值{target}不在数组中')

