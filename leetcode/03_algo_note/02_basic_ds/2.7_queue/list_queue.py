""" 链式队列实现
"""
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

class Queue:
    """ 顺序队列实现
    """
    def __init__(self) -> None:
        """ 初始化空队列
        """
        head = None(0)
        self.front = head
        self.rear = head

    def is_empty(self):
        """ 判断队列是否为空
        """
        return self.front == self.rear
    
    def enqueue(self, value) :
        """ 入队操作
        """
        node = Node(value)
        self.rear.next = node
        self.rear = node
    
    def dequeue(self):
        """ 出队操作
        """
        if self.is_empty():
            raise Exception('Queue is empty')
        else:
            node = self.front.next
            self.front.next = node.next
            if self.rear == node:
                self.rear = self.front
            value = node.value
            del node
            return value
        
    def front_value(self):
        """ 获取队头元素
        """
        if self.is_empty():
            raise Exception('Queue is empty')
        else:
            return self.front.next.value
    
    def rear_value(self):
        """ 获取队尾元素
        """
        if self.is_empty():
            raise Exception('Queue is empty')
        else:
            return self.rear.value


