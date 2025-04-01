from multiprocessing import Process

age = 18

def func():
    global age
    age = 16

if __name__ == '__main__':
    p = Process(target=func)
    p.start()
    p.join()
    print(age) #子进程不会修改主进程数据
    

