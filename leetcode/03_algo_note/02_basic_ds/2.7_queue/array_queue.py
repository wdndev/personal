""" 顺序队列实现
"""

class Queue:
    """ 顺序队列实现
    """
    def __init__(self, size=100) -> None:
        """ 初始化空队列
        """
        self.size = size
        self.queue = [None for _ in range(size)]
        self.front = -1
        self.rear = -1

    def is_empty(self):
        """ 判断队列是否为空
        """
        return self.front == self.rear
    
    def is_full(self):
        """ 判断队列是否已满
        """
        return self.rear + 1 == self.size
    
    def enqueue(self, value) :
        """ 入队操作
        """
        if self.is_full():
            raise Exception('Queue is full')
        else:
            self.rear += 1
            self.queue[self.rear] = value
    
    def dequeue(self):
        """ 出队操作
        """
        if self.is_empty():
            raise Exception('Queue is empty')
        else:
            self.front += 1
            return self.queue[self.front]
        
    def front_value(self):
        """ 获取队头元素
        """
        if self.is_empty():
            raise Exception('Queue is empty')
        else:
            return self.queue[self.front + 1]
    
    def rear_value(self):
        """ 获取队尾元素
        """
        if self.is_empty():
            raise Exception('Queue is empty')
        else:
            return self.queue[self.rear]


