"""
当多个进程操作同一份数据的时候，会出现数据错乱的问题，解决方法就是加锁处理
把并发变成串行，虽然牺牲了运行效率，但是保证了数据安全
加锁只能在争抢数据的时候加

"""

#模拟12306抢票，查询时有票，购买时无票

from multiprocessing import Process, Lock
import time
import json
import random



def search_ticket(name):
    #读取，查询车票数量
    with open('tickets', 'r', encoding='utf-8') as f:
        dic = json.load(f)
    print(f'用户{name}查询余票，{dic.get("tickets_num")}')

def buy_ticket(name):
    with open('tickets', 'r', encoding='utf-8') as f:
        dic = json.load(f)

        time.sleep(random.randint(1, 5))
        if dic.get("tickets_num") > 0:
            dic["tickets_num"] -= 1
            with open('tickets', 'w', encoding='utf-8') as f:
                json.dump(dic, f)
            print(f'用户{name}买票成功')

        else:
            print(f'余票不足，用户{name}买票失败')

def task(name, mutex):
    search_ticket(name)
    mutex.acquire()       #加锁处理 争抢数据
    buy_ticket(name)
    mutex.release()       #释放锁

if __name__ == '__main__':
    mutex = Lock()
    for i in range(1, 9):
        p = Process(target=task, args=(i, mutex))
        p.start()

