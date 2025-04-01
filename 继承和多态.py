'''
面向对象的编程语言支持在已有类的基础上创建新类，以减少重复代码的编写
提供继承信息的类叫做父类，得到继承信息的类叫子类

version：1.0
'''

class Person:
    '''人'''
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def eat(self):
        print(f'{self.name} 正在吃饭！')

    def sleep(self):
        print(f'{self.name} 正在睡觉！')

class Student(Person):
    '''学生'''

    def __init__(self, name, age):
        super().__init__(name, age)  #调用父类的初始化方法

    def study(self, course_name):
        print(f'{self.name}正在学习{course_name}')

class Teacher(Person):
    '''老师'''

    def __init__(self, name, age, title):
        super().__init__(name, age)
        self.title = title

    def teach(self, course_name):
        print(f'{self.name}{self.title}正在讲解{course_name}')


stu1 = Student('路人甲', 22)
stu2 = Student('路人乙', 33)
tea1 = Teacher('张飞', 28, '教授')

stu1.eat()
stu2.sleep()
tea1.eat()
stu1.study('python')
stu2.study('c++')
tea1.teach('gogogo')
