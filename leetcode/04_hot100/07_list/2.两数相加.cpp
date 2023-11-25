/*
 * @lc app=leetcode.cn id=2 lang=cpp
 *
 * [2] 两数相加
 *
 * https://leetcode.cn/problems/add-two-numbers/description/
 *
 * algorithms
 * Medium (42.92%)
 * Likes:    10222
 * Dislikes: 0
 * Total Accepted:    1.9M
 * Total Submissions: 4.5M
 * Testcase Example:  '[2,4,3]\n[5,6,4]'
 *
 * 给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
 * 
 * 请你将两个数相加，并以相同形式返回一个表示和的链表。
 * 
 * 你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：l1 = [2,4,3], l2 = [5,6,4]
 * 输出：[7,0,8]
 * 解释：342 + 465 = 807.
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：l1 = [0], l2 = [0]
 * 输出：[0]
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
 * 输出：[8,9,9,9,0,0,0,1]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 每个链表中的节点数在范围 [1, 100] 内
 * 0 
 * 题目数据保证列表表示的数字不含前导零
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
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        // 头节点
        ListNode* head = new ListNode();
        head->next = nullptr;
        ListNode* head_ptr = head;

        // 两个链表指针
        ListNode* list1_ptr = l1;
        ListNode* list2_ptr = l2;

        // 进位标志
        int carry_bit = 0;

        while (list1_ptr != nullptr || list2_ptr != nullptr) {
            int n1 = list1_ptr ? list1_ptr->val : 0;
            int n2 = list2_ptr ? list2_ptr->val : 0;
            int sum = n1 + n2 + carry_bit;

            // 加入新建的链表中
            ListNode* tmp_node = new ListNode(sum % 10);
            head_ptr->next = tmp_node;
            head_ptr = head_ptr->next;

            // 计算进位
            carry_bit = sum / 10;

            // 更新链表
            if (list1_ptr != nullptr) {
                list1_ptr = list1_ptr->next;
            }

            if (list2_ptr != nullptr) {
                list2_ptr = list2_ptr->next;
            }
        }

        // 如果还有进位，增加结点
        if (carry_bit > 0) {
            ListNode* tmp_node = new ListNode(carry_bit);
            head_ptr->next = tmp_node;
        }

        return head->next;
    }
};
// @lc code=end

