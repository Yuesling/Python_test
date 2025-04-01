'''
定义一个三角形类，通过传入的三条边的长度来构造三角形，并提供计算周长和面积的方法
计算周长和面积为三角形对象的方法
可以先创建一个类方法或静态方法来判断是否是三角形

version = 1.0
'''

class Triangle(object):

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c


    @staticmethod
    def is_vaild(a, b, c):
        '''判断三条边能否构成三角形(静态方法)'''
        return a + b > c and b + c > a and c + a > b

    @property
    def perimeter(self):
        '''计算周长'''
        return self.a + self.b + self.c
    @property
    def area(self):
        '''计算面积'''
        p = self.perimeter / 2
        return (p * (p - self.a) * (p - self.b) * (p - self.c)) ** 0.5

triangle = Triangle(1, 4, 5)
print(triangle.area)
print(triangle.perimeter)

