""" 二叉树遍历
"""

class TreeNode:
    def __init__(self, val, left=None, right=None) -> None:
        self.val = val
        self.left = left
        self.right = right

class Preorder:
    """ 二叉树前序遍历
    """
    def preorder_recu(self, root : TreeNode):
        """ 使用递归实现
        """
        ans = []

        def preorder(root):
            if not root:
                return
            ans.append(root.val)
            preorder(root.left)
            preorder(root.right)

        preorder(root)
        return ans
    
    def preorder_stack(self, root : TreeNode):
        """ 使用显式栈实现
        """
        if not root:
            return []
        ans = []
        stack = [root]

        while stack:                        # 栈不为空
            node = stack.pop()              # 弹出根节点
            ans.append(node.val)            # 访问根节点
            if node.right:
                stack.append(node.right)    # 右子树入栈
            if node.left:
                stack.append(node.left)     # 左子树入栈

        return ans

class Inorder:
    """ 二叉树中序遍历
    """
    def inorder_recu(self, root : TreeNode):
        """ 使用递归实现
        """
        ans = []

        def inorder(root):
            if not root:
                return
            inorder(root.left)
            ans.append(root.val)
            inorder(root.right)

        inorder(root)
        return ans
    
    def inorder_stack(self, root : TreeNode):
        """ 使用显式栈实现
        """
        if not root:                # 二叉树为空直接返回
            return []
        
        ans = []
        stack = []

        while root or stack:        # 根节点或栈不为空
            while root:             
                stack.append(root)  # 将当前树的根节点入栈
                root = root.left    # 找到最左侧节点
            
            node = stack.pop()      # 遍历到最左侧，当前节点无左子树时，将最左侧节点弹出
            ans.append(node.val)    # 访问该节点
            root = node.right       # 尝试访问该节点的右子树
        return ans

class Postorder:
    """ 二叉树后序遍历
    """
    def postorder_recu(self, root : TreeNode):
        """ 使用递归实现
        """
        ans = []

        def postorder(root):
            if not root:
                return
            postorder(root.left)
            postorder(root.right)
            ans.append(root.val)

        postorder(root)
        return ans
    
    def postorder_stack(self, root : TreeNode):
        """ 使用显式栈实现
        """
        ans = []
        stack = []
        prev = None                 # 保存前一个访问的节点，用于确定当前节点的右子树是否访问完毕
        
        while root or stack:        # 根节点或栈不为空
            while root:
                stack.append(root)  # 将当前树的根节点入栈
                root = root.left    # 继续访问左子树，找到最左侧节点

            node = stack.pop()      # 遍历到最左侧，当前节点无左子树时，将最左侧节点弹出

            # 如果当前节点无右子树或者右子树访问完毕
            if not node.right or node.right == prev:
                ans.append(node.val)# 访问该节点
                prev = node         # 记录前一节点
                root = None         # 将当前根节点标记为空
            else:
                stack.append(node)  # 右子树尚未访问完毕，将当前节点重新压回栈中
                root = node.right   # 继续访问右子树
                
        return ans


class LevelOrder:
    def levelorder(self, root: TreeNode):
        """ 队列实现层序遍历
        """
        if not root:
            return []
        queue = [root]
        order = []
        while queue:
            level = []
            size = len(queue)
            for _ in range(size):
                curr = queue.pop(0)
                level.append(curr.val)
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)
            if level:
                order.append(level)
        return order
