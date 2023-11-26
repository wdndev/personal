/*
 * @lc app=leetcode.cn id=23 lang=cpp
 *
 * [23] 合并 K 个升序链表
 *
 * https://leetcode.cn/problems/merge-k-sorted-lists/description/
 *
 * algorithms
 * Hard (58.65%)
 * Likes:    2708
 * Dislikes: 0
 * Total Accepted:    730.7K
 * Total Submissions: 1.2M
 * Testcase Example:  '[[1,4,5],[1,3,4],[2,6]]'
 *
 * 给你一个链表数组，每个链表都已经按升序排列。
 * 
 * 请你将所有链表合并到一个升序链表中，返回合并后的链表。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 输入：lists = [[1,4,5],[1,3,4],[2,6]]
 * 输出：[1,1,2,3,4,4,5,6]
 * 解释：链表数组如下：
 * [
 * ⁠ 1->4->5,
 * ⁠ 1->3->4,
 * ⁠ 2->6
 * ]
 * 将它们合并到一个有序链表中得到。
 * 1->1->2->3->4->4->5->6
 * 
 * 
 * 示例 2：
 * 
 * 输入：lists = []
 * 输出：[]
 * 
 * 
 * 示例 3：
 * 
 * 输入：lists = [[]]
 * 输出：[]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * k == lists.length
 * 0 <= k <= 10^4
 * 0 <= lists[i].length <= 500
 * -10^4 <= lists[i][j] <= 10^4
 * lists[i] 按 升序 排列
 * lists[i].length 的总和不超过 10^4
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
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        ListNode* ans = nullptr;

        for (int i = 0; i < lists.size(); i++) {
            ans = this->mergeTwoLists(ans, lists[i]);
        }

        return ans;
    }

private:
    // 合并两个有序链表
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode* list1_ptr = list1;
        ListNode* list2_ptr = list2;

        ListNode* head = new ListNode();
        head->next = nullptr;
        ListNode* head_ptr = head;

        while (list1_ptr != nullptr && list2_ptr != nullptr) {
            if (list1_ptr->val <= list2_ptr->val) {
                head_ptr->next = list1_ptr;
                head_ptr = head_ptr->next;
                list1_ptr = list1_ptr->next;
            } else if (list1_ptr->val > list2_ptr->val) {
                head_ptr->next = list2_ptr;
                head_ptr = head_ptr->next;
                list2_ptr = list2_ptr->next;
            }
        }

        // 如果 a b两个中其中一个没有结束，接在后面
        // head_ptr->next = list1_ptr != nullptr ? list1_ptr : list2_ptr;

        while (list1_ptr != nullptr) {
            head_ptr->next = list1_ptr;
            head_ptr = head_ptr->next;
            list1_ptr = list1_ptr->next;
        }

        while (list2_ptr != nullptr) {
            head_ptr->next = list2_ptr;
            head_ptr = head_ptr->next;
            list2_ptr = list2_ptr->next;
        }

        return head->next;
    }
};
// @lc code=end

