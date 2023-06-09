from multiprocessing import Process
import os, time

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    info('function f')
    print('hello', name)
    time.sleep(1)

def g(name):
    info('function g')
    time.sleep(2)
    print('hello again', name)
    return "I ran!"

if __name__ == '__main__':
    info('main line')
    
    p = Process(target=f, args=('bob',))
    q = Process(target=g, args=('jon',))
    p.start()
    q.start()
    print("waiting")
    p.join()
    print("still waiting")
    q.join()
    print("done waiting") 