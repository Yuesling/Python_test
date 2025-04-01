# 对字符串做全排列
# 例如：abc 全排列工6种结果 abc acb bac bca cab cba


s = 'abcd'

l = list(s)

def permutation(l, level):
    if level == len(l):
        print(l)
    for i in range(level, len(l)):
        l[level], l[i] = l[i], l[level]
        permutation(l, level + 1)
        l[level], l[i] = l[i], l[level]

permutation(l, 0)
