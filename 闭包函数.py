
def func1(x):
    def func2():
        print(x)
    return func2

res = func1(20)
res()
