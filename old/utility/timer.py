import numpy as np
import time

class Timer :
    """ 记录多次运行时间
    """
    def __init__(self) -> None:
        self.times = []
        # self.start()

    def start(self):
        """ 启动计时器
        """
        self.tik = time.time()

    def stop(self):
        """ 停止计时器，并将时间记录在列表中
        """
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def clear(self):
        """ 清除计时器内记录
        """
        self.times.clear()

    def avg(self):
        """ 返回平均时间
        """
        return sum(self.times) / len(self.times)

    def sum(self):
        """ 返回总时间
        """
        return sum(self.times)
    
    def cumsum(self):
        """ 返回累计时间
        """
        return np.array(self.times).cumsum().tolist()