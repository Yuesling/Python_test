from multiprocessing import Process
import time

def func(name):
    print(f'{name}开始')
    time.sleep(5)
    print(f'{name}任务结束')

if __name__ == '__main__':

    p = Process(target=func, args=('python',))
    p.start()
    p.join()    #让主进程在子进程执行完成后执行
    print('hello')

