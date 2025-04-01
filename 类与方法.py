class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def study(self, course_name):
        print(f'{self.name}{self.age}岁，正在学习{course_name}')

stu = Student('wangdachui', 18)
stu.study('python')
