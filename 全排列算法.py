# 对字符串做全排列
# 例如：abc 全排列工6种结果 abc acb bac bca cab cba


s = 'abcd'

l = list(s)

# def permutation(l, level):
#     if level == len(l):
#         print(l)
#     for i in range(level, len(l)):
#         l[level], l[i] = l[i], l[level]
#         permutation(l, level + 1)
#         l[level], l[i] = l[i], l[level]
#
# permutation(l, 0)


# 方法一：暴力for循环
# for循环，每次检查是否重复字符
# 仅适用固定长度，不推荐使用
# for i in s:
#     for j in s:
#         if j != i:
#             for k in s:
#                 if k != j and k != i:
#                     for n in s:
#                         if n != k and n != j and n != i:
#                             print(i + j + k + n)

# 方法二：使用python标准库 itertools.permutations
# permutations生成所有的排列组合
# 返回的是个迭代器

# from itertools import permutations
#
# result = permutations(s)
# result_list = [''.join(p) for p in result]
# print(result_list)

# 方法三：DFS+回溯(经典算法)
# 使用深度优先遍历 和回溯解决

# def permute(s, path='', visited=None, result=None):
#     """
#     :param s:需要排列的字符串
#     :param path:当前递归路径已生成的部分
#     :param visited:标记数组，用于记录哪些字符被使用
#     :param result:排列结果
#     :return:返回排列结果
#     """
#
#     #初始化访问标记数组（记录哪些字符已经被使用）
#     if visited is None:
#         visited = [False] * len(s)        #[False, False, False, False]
#
#     #初始化结果列表：
#     if result is None:
#         result = []
#
#     #如果当前路径长度等于原字符串长度，说明已经生成一个排列
#     if len(path) == len(s):
#         result.append(path)
#         return
#
#     #遍历字符串中的每个字符
#     for i in range(len(s)):
#         if not visited[i]:                #如果该字符未被使用，则标记改为True
#             visited[i] = True
#             permute(s, path + s[i], visited, result)  #采用递归，将该字符加入路径，继续生成其他排列
#             visited[i] = False            #回溯，重置标记，供其他路径使用
#
#     return result
#
#
# print(permute(s))

# 方法四：堆算法

def head_permutations(data, n):
    # 如果只剩一个字符，直接输出当前排列
    if n == 1:
        print(''.join(data))
    else:
        for i in range(n):
            # 递归生成前n-1个字符的排列
            head_permutations(data, n - 1)
            # 根据n的奇偶性交换字符位置
            if n % 2 == 0:
                data[i], data[n-1] = data[n-1], data[i]
            else:
                data[0], data[n-1] = data[n-1], data[0]

head_permutations(list(s), len(s))


