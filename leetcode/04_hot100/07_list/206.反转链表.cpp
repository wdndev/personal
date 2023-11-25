/*
 * @lc app=leetcode.cn id=206 lang=cpp
 *
 * [206] 反转链表
 *
 * https://leetcode.cn/problems/reverse-linked-list/description/
 *
 * algorithms
 * Easy (73.67%)
 * Likes:    3395
 * Dislikes: 0
 * Total Accepted:    1.6M
 * Total Submissions: 2.2M
 * Testcase Example:  '[1,2,3,4,5]'
 *
 * 给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：head = [1,2,3,4,5]
 * 输出：[5,4,3,2,1]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：head = [1,2]
 * 输出：[2,1]
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：head = []
 * 输出：[]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 链表中节点的数目范围是 [0, 5000]
 * -5000 
 * 
 * 
 * 
 * 
 * 进阶：链表可以选用迭代或递归方式完成反转。你能否用两种方法解决这道题？
 * 
 * 
 * 
 */

// @lc code=start
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        // 增加头结点，方便插入
        ListNode* head_point = new ListNode();
        head_point->next = nullptr;
        ListNode* node_ptr = head;
        while (node_ptr != nullptr) {
            // 暂存下一个结点
            ListNode* next_node = node_ptr->next;
            // step1:先将插入结点指针修改为头节点的后继
            node_ptr->next = head_point->next;
            // step2： 再将头节点L的后继更新为新插入结点
            head_point->next = node_ptr;

            // 更新下一个节点
            node_ptr = next_node;
        }
        // 注意，增加了头节点，需要去掉
        return head_point->next;
    }
};
// @lc code=end

