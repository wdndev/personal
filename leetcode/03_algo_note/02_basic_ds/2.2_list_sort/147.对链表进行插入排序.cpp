/*
 * @lc app=leetcode.cn id=147 lang=cpp
 *
 * [147] 对链表进行插入排序
 *
 * https://leetcode.cn/problems/insertion-sort-list/description/
 *
 * algorithms
 * Medium (69.53%)
 * Likes:    651
 * Dislikes: 0
 * Total Accepted:    159.5K
 * Total Submissions: 229.3K
 * Testcase Example:  '[4,2,1,3]'
 *
 * 给定单个链表的头 head ，使用 插入排序 对链表进行排序，并返回 排序后链表的头 。
 * 
 * 插入排序 算法的步骤:
 * 
 * 
 * 插入排序是迭代的，每次只移动一个元素，直到所有元素可以形成一个有序的输出列表。
 * 每次迭代中，插入排序只从输入数据中移除一个待排序的元素，找到它在序列中适当的位置，并将其插入。
 * 重复直到所有输入数据插入完为止。
 * 
 * 
 * 
 * 下面是插入排序算法的一个图形示例。部分排序的列表(黑色)最初只包含列表中的第一个元素。每次迭代时，从输入数据中删除一个元素(红色)，并就地插入已排序的列表中。
 * 
 * 对链表进行插入排序。
 * 
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 
 * 
 * 输入: head = [4,2,1,3]
 * 输出: [1,2,3,4]
 * 
 * 示例 2：
 * 
 * 
 * 
 * 
 * 输入: head = [-1,5,3,4,0]
 * 输出: [-1,0,3,4,5]
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 
 * 
 * 列表中的节点数在 [1, 5000]范围内
 * -5000 <= Node.val <= 5000
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
    ListNode* insertionSortList(ListNode* head) {
        if (head == nullptr || head->next == nullptr) {
            return head;
        }
        ListNode* dummy_head = new ListNode(-1);
        dummy_head->next = head;

        // 已排好序的部分的最后一个节点
        ListNode* sort_list = head;
        // 待插入元素
        ListNode* curr_node = head->next;

        while (curr_node != nullptr) {
            if (sort_list->val <= curr_node->val) {
                // 将curr_node 插入到 sortlist 之后
                sort_list = sort_list->next;
            } else {
                ListNode* prev = dummy_head;
                while (prev->next->val <= curr_node->val) {
                    prev = prev->next;
                }
                // 将 curr_node 插入到链表中
                sort_list->next = curr_node->next;
                curr_node->next = prev->next;
                prev->next = curr_node;
            }
            curr_node = sort_list->next;
        }

        return dummy_head->next;
    }
};
// @lc code=end

