/*
 * @lc app=leetcode.cn id=21 lang=cpp
 *
 * [21] 合并两个有序链表
 *
 * https://leetcode.cn/problems/merge-two-sorted-lists/description/
 *
 * algorithms
 * Easy (66.19%)
 * Likes:    3403
 * Dislikes: 0
 * Total Accepted:    1.6M
 * Total Submissions: 2.4M
 * Testcase Example:  '[1,2,4]\n[1,3,4]'
 *
 * 将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：l1 = [1,2,4], l2 = [1,3,4]
 * 输出：[1,1,2,3,4,4]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：l1 = [], l2 = []
 * 输出：[]
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：l1 = [], l2 = [0]
 * 输出：[0]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 两个链表的节点数目范围是 [0, 50]
 * -100 
 * l1 和 l2 均按 非递减顺序 排列
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
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode* dummy_head = new ListNode(-1);
        dummy_head->next = nullptr;


        ListNode* dummy_ptr = dummy_head;
        ListNode* list1_ptr = list1;
        ListNode* list2_ptr = list2;

        while (list1_ptr!= nullptr && list2_ptr!= nullptr) {
            if (list1_ptr->val <= list2_ptr->val) {
                ListNode* tmp_node = list1_ptr->next;
                list1_ptr->next = dummy_ptr->next;
                dummy_ptr->next = list1_ptr;
                list1_ptr = tmp_node;
            } else {
                ListNode* tmp_node = list2_ptr->next;
                list2_ptr->next = dummy_ptr->next;
                dummy_ptr->next = list2_ptr;
                list2_ptr = tmp_node;
            }
            dummy_ptr = dummy_ptr->next;
        }

        // 如果 a b两个中其中一个没有结束，接在后面
        // head_ptr->next = list1_ptr != nullptr ? list1_ptr : list2_ptr;

        while (list1_ptr != nullptr) {
            dummy_ptr->next = list1_ptr;
            dummy_ptr = dummy_ptr->next;
            list1_ptr = list1_ptr->next;
        }

        while (list2_ptr != nullptr) {
            dummy_ptr->next = list2_ptr;
            dummy_ptr = dummy_ptr->next;
            list2_ptr = list2_ptr->next;
        }

        return dummy_head->next;
    }
};
// @lc code=end

