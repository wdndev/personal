#
# @lc app=leetcode.cn id=148 lang=python3
#
# [148] 排序链表
#
# https://leetcode.cn/problems/sort-list/description/
#
# algorithms
# Medium (65.52%)
# Likes:    2205
# Dislikes: 0
# Total Accepted:    457.1K
# Total Submissions: 697.8K
# Testcase Example:  '[4,2,1,3]'
#
# 给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。
# 
# 
# 
# 
# 
# 
# 示例 1：
# 
# 
# 输入：head = [4,2,1,3]
# 输出：[1,2,3,4]
# 
# 
# 示例 2：
# 
# 
# 输入：head = [-1,5,3,4,0]
# 输出：[-1,0,3,4,5]
# 
# 
# 示例 3：
# 
# 
# 输入：head = []
# 输出：[]
# 
# 
# 
# 
# 提示：
# 
# 
# 链表中节点的数目在范围 [0, 5 * 10^4] 内
# -10^5 <= Node.val <= 10^5
# 
# 
# 
# 
# 进阶：你可以在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序吗？
# 
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        return self.merge_sort(head)

    def bubble_sort(self, head : ListNode) -> ListNode:
        """ 冒泡排序
        """
        # node_i : 用于控制外循环的次数
        # node_j : 用于控制内循环的次数
        # tail : 循环结束位置
        node_i = head
        tail = None
        # 外层训练次数：链表节点个数
        while node_i:
            node_j = head
            while node_j and node_j.next != tail:
                if node_j.val > node_j.next.val:
                    # 交换两个节点的值
                    node_j.val, node_j.next.val = node_j.next.val, node_j.val
                node_j = node_j.next
            # 一次循环之后，将 tail 移动到 node_j 所在位置。相当于 tail 向左移动了一位。
            # 尾指针向前移动一位，此时尾指针右侧为排好序的数组
            tail = node_j
            node_i = node_i.next
        return head
    
    def insert_sort(self, head: ListNode) -> ListNode:
        """ 插入排序
        """
        if not head or not head.next:
            return head
        
        dummy_head = ListNode(-1)
        dummy_head.next = head
        # 已排好序部分的最后一个节点
        sorted_list = head 
        # 待插入元素
        curr_node = head.next

        while curr_node:
            if sorted_list.val <= curr_node.val:
                # 将 cur 插入到 sorted_list 之后
                sorted_list = sorted_list.next
            else:
                prev = dummy_head
                while prev.next.val <= curr_node.val:
                    prev = prev.next
                # 将 cur 到链表中间
                sorted_list.next = curr_node.next
                curr_node.next = prev.next
                prev.next = curr_node
            curr_node = sorted_list.next

        return dummy_head.next
    

    def merge(self, left, right):
        """ 归并排序, 合并
        """
        dummy_head = ListNode(-1)
        curr_node = dummy_head
        while left and right:
            if left.val <= right.val:
                curr_node.next = left
                left = left.next
            else:
                curr_node.next = right
                right = right.next
            curr_node = curr_node.next

        if left:
            curr_node.next = left
        if right:
            curr_node.next = right
        
        return dummy_head.next
    
    def merge_sort(self, head: ListNode) -> ListNode:
        """ 归并排序，分割
        """
        if not head or not head.next:
            return head
        
        # 快慢指针找到中心链节点
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        # 断开左右链节点
        left_head = head 
        right_head = slow.next
        slow.next = None

        # 归并
        return self.merge(self.merge_sort(left_head), self.merge_sort(right_head))

    def partition(self, left: ListNode, right: ListNode):
        """ 快排划分
        """
        # 左闭右开，区间没有元素或者只有一个元素，直接返回第一个节点
        if left == right or left.next == right:
            return left
        # 选择头节点为基准节点
        pivot = left.val
        # 使用 node_i, node_j 双指针，保证 node_i 之前的节点值都小于基准节点值，
        # node_i 与 node_j 之间的节点值都大于等于基准节点值
        node_i, node_j = left, left.next
        
        while node_j != right:
            # 发现一个小与基准值的元素
            if node_j.val < pivot:
                # 因为 node_i 之前节点都小于基准值，
                # 所以先将 node_i 向右移动一位（此时 node_i 节点值大于等于基准节点值）
                node_i = node_i.next
                # 将小于基准值的元素 node_j 与当前 node_i 换位，
                # 换位后可以保证 node_i 之前的节点都小于基准节点值
                node_i.val, node_j.val = node_j.val, node_i.val
            node_j = node_j.next
        # 将基准节点放到正确位置上
        node_i.val, left.val = left.val, node_i.val
        return node_i
        
    def quick_sort(self, left: ListNode, right: ListNode):
        """ 快速排序
        """
        if left == right or left.next == right:
            return left
        pi = self.partition(left, right)
        self.quickSort(left, pi)
        self.quickSort(pi.next, right)
        return left
        

        







# @lc code=end

