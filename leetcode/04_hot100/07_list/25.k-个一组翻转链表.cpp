/*
 * @lc app=leetcode.cn id=25 lang=cpp
 *
 * [25] K 个一组翻转链表
 *
 * https://leetcode.cn/problems/reverse-nodes-in-k-group/description/
 *
 * algorithms
 * Hard (67.70%)
 * Likes:    2201
 * Dislikes: 0
 * Total Accepted:    514.6K
 * Total Submissions: 760K
 * Testcase Example:  '[1,2,3,4,5]\n2'
 *
 * 给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。
 * 
 * k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
 * 
 * 你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：head = [1,2,3,4,5], k = 2
 * 输出：[2,1,4,3,5]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 
 * 
 * 输入：head = [1,2,3,4,5], k = 3
 * 输出：[3,2,1,4,5]
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 链表中的节点数目为 n
 * 1 <= k <= n <= 5000
 * 0 <= Node.val <= 1000
 * 
 * 
 * 
 * 
 * 进阶：你可以设计一个只用 O(1) 额外内存空间的算法解决此问题吗？
 * 
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
    ListNode* reverseKGroup(ListNode* head, int k) {
        ListNode* head_node = new ListNode(0);
        head_node->next = head;
        ListNode* pre = head_node;

        while (head != nullptr) {
            ListNode* tail = pre;

            // 查看剩余部分长度是否大于等于k
            for (int i = 0; i < k; i++) {
                tail = tail->next;
                if (tail == nullptr) {
                    return head_node->next;
                }
            }

            ListNode* next = tail->next;
            // 这里是 C++17 的写法，也可以写成
            // pair<ListNode*, ListNode*> result = myReverse(head, tail);
            // head = result.first;
            // tail = result.second;
            std::tie(head, tail) = this->reverse_list(head, tail);

            // 把子链重新接回原链表
            pre->next = head;
            tail->next = next;
            pre = tail;
            head = tail->next;
        }

        return head_node->next;
    }

private:
    // 翻转一个子链表，并且返回新的头和尾
    std::pair<ListNode*, ListNode*> reverse_list(ListNode* head, ListNode*tail) {
        ListNode* prev = tail->next;
        ListNode* p = head;

        while (prev != tail) {
            ListNode* tmp = p->next;
            p->next = prev;

            prev = p;
            p = tmp;
        }

        return {tail, head};
    }
};
// @lc code=end

