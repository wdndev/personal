/*
 * @lc app=leetcode.cn id=148 lang=cpp
 *
 * [148] 排序链表
 *
 * https://leetcode.cn/problems/sort-list/description/
 *
 * algorithms
 * Medium (65.54%)
 * Likes:    2161
 * Dislikes: 0
 * Total Accepted:    448.1K
 * Total Submissions: 683.9K
 * Testcase Example:  '[4,2,1,3]'
 *
 * 给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。
 * 
 * 
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：head = [4,2,1,3]
 * 输出：[1,2,3,4]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：head = [-1,5,3,4,0]
 * 输出：[-1,0,3,4,5]
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
 * 链表中节点的数目在范围 [0, 5 * 10^4] 内
 * -10^5 <= Node.val <= 10^5
 * 
 * 
 * 
 * 
 * 进阶：你可以在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序吗？
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
    ListNode* sortList1(ListNode* head) {
        // 一直扫描至倒数第二个节点，判断每个结点与后继
        while (1) {
            ListNode* node_ptr = head;
            // 数据交换标志位，
            int is_exchange_flag = 0;
            // 访问每一对前去和后继
            while (node_ptr != nullptr && node_ptr->next != nullptr) {
                // 判断是否逆序，如果逆序，交换元素
                if (node_ptr->val > node_ptr->next->val) {
                    int tmp = node_ptr->val;
                    node_ptr->val = node_ptr->next->val;
                    node_ptr->next->val = tmp;
                    is_exchange_flag = 1;
                }
                // 指针移动到下一个
                node_ptr = node_ptr->next;
            }
            if (is_exchange_flag == 0) {
                break;
            }
        }

        return head;
    }

    ListNode* sortList(ListNode* head) {
        if(!head) return nullptr;
        multiset<int> set;
        ListNode* ptr = head;
        while(ptr)
        {
            set.insert(ptr->val);
            ptr = ptr->next;
        }
        ptr = head;
        for(auto it=set.begin(); it!=set.end(); it++) {
        ptr->val = *it;
            ptr = ptr->next;
      }
        return head;
    }
};
// @lc code=end

