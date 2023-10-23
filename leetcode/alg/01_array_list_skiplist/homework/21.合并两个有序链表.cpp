/*
 * @lc app=leetcode.cn id=21 lang=cpp
 *
 * [21] 合并两个有序链表
 *
 * https://leetcode.cn/problems/merge-two-sorted-lists/description/
 *
 * algorithms
 * Easy (66.22%)
 * Likes:    3322
 * Dislikes: 0
 * Total Accepted:    1.5M
 * Total Submissions: 2.3M
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
        if (list1 == nullptr && list2 == nullptr) {
            return nullptr;
        }
        ListNode* node_ptr1 = list1;
        ListNode* node_ptr2 = list2;

        ListNode* ans = new ListNode(-1);
        ListNode* ans_ptr = ans;

        while(node_ptr1 != nullptr && node_ptr2 != nullptr) {
            if (node_ptr1->val <= node_ptr2->val) {
                ans_ptr->next = node_ptr1;
                ans_ptr = ans_ptr->next;
                node_ptr1 = node_ptr1->next;
            } else {
                ans_ptr->next = node_ptr2;
                ans_ptr = ans_ptr->next;
                node_ptr2 = node_ptr2->next;
            }
        }

        // 如果 a b两个中其中一个没有结束，接在后面
        ans_ptr->next = node_ptr1 != nullptr ? node_ptr1 : node_ptr2;

        // while (node_ptr1 != nullptr) {
        //     ans_ptr->next = node_ptr1;
        //     ans_ptr = ans_ptr->next;
        //     node_ptr1 = node_ptr1->next;
        // }

        // while (node_ptr2 != nullptr) {
        //     ans_ptr->next = node_ptr2;
        //     ans_ptr = ans_ptr->next;
        //     node_ptr2 = node_ptr2->next;
        // }


        return ans->next;
        
    }
};
// @lc code=end

