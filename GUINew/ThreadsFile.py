from PyQt5.QtCore import QRunnable, QThreadPool, QObject

class MyThread(QRunnable):
    
    def __init__(self,function,args):
        super().__init__()
        self.function=function
        self.args=args
        
    def run(self):
        self.function(*self.args)

class MyThreadManager(QObject):
    def __init__(self):
        super().__init__()
        self.thread_pool=QThreadPool()
        
    def start_thread(self,function,args):
        handle=MyThread(function=function,args=args)
        self.thread_pool.start(handle)