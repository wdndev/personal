# 链式栈实现
class Node:
    """ 链表结点
    """
    def __init__(self, value) -> None:
        self.value = value
        self.next = None

class Stack:
    """ 链式栈实现
    """
    def __init__(self) -> None:
        """ 初始化空栈
        """
        self.top = None

    def is_empty(self):
        """ 判断栈是否为空
        """
        return self.top == None
    
    
    def push(self, value):
        """ 入栈操作
        """
        curr_node = Node(value)
        curr_node.next = self.top
        self.top = curr_node

    def pop(self):
        """ 出栈
        """
        if self.is_empty():
            raise Exception("stack is empty")
        else:
            curr_node = self.top
            self.top = self.top.next
            del curr_node

    def peek(self):
        """ 获取栈顶元素
        """
        if self.is_empty():
            raise Exception("stack is empty")
        else:
            return self.top.value
    


