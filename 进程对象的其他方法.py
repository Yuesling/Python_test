from multiprocessing import process, current_process, Process
import time
import os

#pid(进程号)
def task(name):
    print(f'任务{current_process().pid}正在执行')   #方法一
    print(f'{name}{os.getpid()}执行中')
    print(f'{name}的父进程{os.getppid()}执行中')

    time.sleep(10)

if __name__ == '__main__':
    p = Process(target=task, args=('子进程',))
    p.start()
    p.terminate()  #kill进程
    print(p.is_alive())   #判断进程是否存活
    print('主进程', current_process().pid)
