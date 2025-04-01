def func1(x, y, z):
    print('func1>>>', x, y, z)

def func2(*args, **kwargs):
    func1(*args, **kwargs)


#func2(1, 2, 3)
func2(1, 2, z=3)



