from multiprocessing import Process
import time

class Func(Process):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def run(self):
        print(f'{self.name} start')
        time.sleep(5)
        print(f'{self.name} end')

if __name__ == '__main__':
    p = Func('python')
    p.start()
    print('hello')