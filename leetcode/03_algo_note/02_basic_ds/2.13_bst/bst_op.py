""" 二叉搜索树的基本操作
"""

class TreeNode:
    def __init__(self, val=0, left=None, right=None) -> None:
        self.val = val
        self.left = left
        self.right = right

class BST:
    """ 二叉搜索树
    """
    def search_bst(self, root:TreeNode, val:int) -> TreeNode:
        """ 二叉搜索树查找
        """
        if not root:
            return None
        
        if val == root.val:
            return root
        elif val < root.val:
            return self.search_bst(root.left, val)
        else:
            return self.search_bst(root.right, val)
        
    def insert_bst(self, root:TreeNode, val:int) -> TreeNode:
        """ 二叉搜索值插入
        """
        if root == None:
            return TreeNode(val)
        
        if val < root.val:
            root.left = self.insert_bst(root.left, val)
        if val > root.val:
            root.right = self.insert_bst(root.right, val)
        return root
    
    def build_bst(self, nums:list) -> TreeNode:
        """ 从列表新建二叉树
        """
        root = TreeNode()
        for n in nums:
            self.insert_bst(root, n)
        return root

    def delete_node(self, root:TreeNode, val:int) -> TreeNode:
        """ 删除某个节点
        """
        if not root:
            return root
        
        if root.val > val:
            root.left = self.delete_node(root.left, val)
            return root
        elif root.val < val:
            root.right = self.delete_node(root.right, val)
            return root
        else:
            # 根节点左子树为空，返回右子树
            if not root.left:
                return root.right
            # 根节点右子树为空，返回左子树
            elif not root.right:
                return root.left
            else:
                # 将root节点的左子树挂到右子树上
                ## 1.找到右子树的最小值，
                curr_node = root.right
                while curr_node.left:
                    curr_node = curr_node.left
                ## 2.将根节点的左子树挂到右子树的最小值
                curr_node.left = root.left

                return root.right 


