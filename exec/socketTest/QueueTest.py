import time
import random
from multiprocessing import Process, JoinableQueue

"""
 # 进程结束步骤 :
    1. 当主进程执行完毕后 会堵塞在 q.join()
    2. 当 q.join() 阻塞结束后 主进程也就结束了
    3. 因为 守护进程的原因,主进程结束后,子进程也会跟着结束
"""


def consumer(name, q):
    while True:
        food = q.get()
        if food is None:
            print('zzz..')
            break
        print('\033[31m%s 消费了 %s\033[0m' % (name, food))
        time.sleep(random.randint(1, 2))


def producer(name, food, q):
    for i in range(3):
        time.sleep(random.randint(1, 2))
        a = '%s上产了%s %s次' % (name, food, i)
        print(a)
        q.put(i)


if __name__ == '__main__':
    q = JoinableQueue()
    p1 = Process(target=producer, args=('q', 'apple', q))  # 生产
    p2 = Process(target=producer, args=('qq', 'pear', q))  # 生产
    c1 = Process(target=consumer, args=('w', q))  # 消费
    c2 = Process(target=consumer, args=('e', q))  # 消费

    p1.start()
    p2.start()
    c1.start()
    c2.start()
    p1.join()
    p2.join()
    q.put(None)
    q.put(None)