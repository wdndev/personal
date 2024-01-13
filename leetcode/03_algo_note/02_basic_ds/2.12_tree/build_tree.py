""" 构造二叉树
"""
class TreeNode:
    def __init__(self, val, left=None, right=None) -> None:
        self.val = val
        self.left = left
        self.right = right


class BuildTree:
    """ 构造二叉树
    """
    def prein_build_tree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        """ 前序中序构造二叉树
        """
        def create_tree(preorder, inorder, n):
            """
                - preorder
                - inorder
                - n : 中序的长度
            """
            if n == 0:
                return None
            
            # 拆分中序序列，以根节点为中心，拆分为左右节点
            k = 0
            while preorder[0] != inorder[k]:
                k += 1
            
            node = TreeNode(inorder[k])
            node.left = create_tree(preorder[1:k + 1], inorder[0, k], k)
            node.right = create_tree(preorder[k + 1:], inorder[k+1:], n-k-1)
            return node
        
        return create_tree(preorder, inorder, len(inorder))
    
    def postin_build_tree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        """ 后序中序构建二叉树
        """
        def create_tree(inorder, postorder, n):
            """
                - preorder
                - inorder
                - n : 中序的长度
            """
            if n == 0:
                return None
            
            # 拆分中序序列，以根节点为中心，拆分为左右节点
            k = 0
            while postorder[n-1] != inorder[k]:
                k += 1
            
            node = TreeNode(inorder[k])
            node.left = create_tree(inorder[0, k], postorder[0:k], k)
            node.right = create_tree(inorder[k+1:n], postorder[k:n-1], n-k-1)
            return node
        
        return create_tree(inorder, postorder, len(postorder))




