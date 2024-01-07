# 顺序栈实现

class Stack:
    """ 顺序栈实现
    """
    def __init__(self, size=100) -> None:
        """ 初始化
        """
        self.stack = []
        self.size = size 
        self.top = -1

    def is_empty(self):
        """ 判断栈是否为空
        """
        return self.top == -1
    
    def is_full(self):
        """ 判断栈是否已满
        """
        return self.top + 1 == self.size
    
    def push(self, value):
        """ 入栈操作
        """
        if self.is_full():
            raise Exception("stack is full")
        else:
            self.stack.append(value)
            self.top += 1

    def pop(self):
        """ 出栈
        """
        if self.is_empty():
            raise Exception("stack is empty")
        else:
            self.stack.pop()
            self.top -= 1

    def peek(self):
        """ 获取栈顶元素
        """
        if self.is_empty():
            raise Exception("stack is empty")
        else:
            return self.stack[self.top]


