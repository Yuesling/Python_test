'''
通过内置函数open来对文件进行操作
    name: 需要操作的文件名
    mode：
        'r':读取(默认)
        'w':写入，会先截断之前的内容
        'x':写入，如果文件已经存在会报错
        'a':追加，将内容写入到文件末尾
        'b':二进制模式
        't':文本模式，默认
        '+':更新，既可以读又可以写
    encoding: 编码格式
'''

#file = open('python.txt', 'r', encoding='utf-8')
#print(file.read())
file = open('python.txt', 'a', encoding='utf-8')
file.write('\nhello world')

# for line in file:
#     print(line)


# lines = file.readlines()
# for line in lines:
#     print(line)
file.close()

