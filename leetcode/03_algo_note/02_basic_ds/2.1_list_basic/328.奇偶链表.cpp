/*
 * @lc app=leetcode.cn id=328 lang=cpp
 *
 * [328] 奇偶链表
 *
 * https://leetcode.cn/problems/odd-even-linked-list/description/
 *
 * algorithms
 * Medium (64.65%)
 * Likes:    761
 * Dislikes: 0
 * Total Accepted:    227.6K
 * Total Submissions: 352.1K
 * Testcase Example:  '[1,2,3,4,5]'
 *
 * 给定单链表的头节点 head ，将所有索引为奇数的节点和索引为偶数的节点分别组合在一起，然后返回重新排序的列表。
 * 
 * 第一个节点的索引被认为是 奇数 ， 第二个节点的索引为 偶数 ，以此类推。
 * 
 * 请注意，偶数组和奇数组内部的相对顺序应该与输入时保持一致。
 * 
 * 你必须在 O(1) 的额外空间复杂度和 O(n) 的时间复杂度下解决这个问题。
 * 
 * 
 * 
 * 示例 1:
 * 
 * 
 * 
 * 
 * 输入: head = [1,2,3,4,5]
 * 输出: [1,3,5,2,4]
 * 
 * 示例 2:
 * 
 * 
 * 
 * 
 * 输入: head = [2,1,3,5,6,4,7]
 * 输出: [2,3,6,7,1,5,4]
 * 
 * 
 * 
 * 提示:
 * 
 * 
 * n ==  链表中的节点数
 * 0 <= n <= 10^4
 * -10^6 <= Node.val <= 10^6
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
    ListNode* oddEvenList(ListNode* head) {
        if (head == nullptr || head->next == nullptr 
            || head->next->next == nullptr) {
             return head;
        }
        // 偶数链表节点头指针
        ListNode* even_head = head->next;
        // 奇数链表节点头指针
        ListNode* odd = head;
        ListNode* even = even_head;
        
        bool is_odd = true;
        ListNode* curr_node = head->next->next;

        while (curr_node != nullptr) {
            // 奇数节点，添加奇数链表
            if (is_odd) {
                odd->next = curr_node;
                odd = curr_node;
            } else {
                // 偶数节点，添加偶数链表
                even->next = curr_node;
                even = curr_node;
            }
            is_odd = !is_odd;
            curr_node = curr_node->next;
        }
        odd->next = even_head;
        even->next = nullptr;

        return head;
    }
};
// @lc code=end

