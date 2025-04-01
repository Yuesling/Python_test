from multiprocessing import Process
import time

def func(name, n):
    print(f'{name}开始')
    time.sleep(n)
    print(f'{name}任务结束')

if __name__ == '__main__':
    start = time.time()
    # p1 = Process(target=func, args=('python1', 1))
    # p2 = Process(target=func, args=('python2', 2))
    # p3 = Process(target=func, args=('python3', 3))
    # p1.start()
    # p2.start()
    # p3.start()
    # p1.join()
    # p2.join()
    # p3.join()
    l = []
    for i in range(1, 4):
        p = Process(target=func, args=(f'python{i}', i))
        p.start()
        l.append(p)
    for p in l:
        p.join()
    print('hello')
    end = time.time()
    print(f'{end - start}')

