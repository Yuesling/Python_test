from multiprocessing import Process
import time

def func(name):
    print(f'{name}开始')
    time.sleep(5)
    print(f'{name}任务结束')

if __name__ == '__main__':

    p = Process(target=func, args=('python',))
    p.daemon = True     #父进程与子进程同时结束 
    p.start()
    print('hello')

