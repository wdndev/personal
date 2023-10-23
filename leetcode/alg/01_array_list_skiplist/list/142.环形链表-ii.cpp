/*
 * @lc app=leetcode.cn id=142 lang=cpp
 *
 * [142] 环形链表 II
 *
 * https://leetcode.cn/problems/linked-list-cycle-ii/description/
 *
 * algorithms
 * Medium (57.80%)
 * Likes:    2369
 * Dislikes: 0
 * Total Accepted:    763.9K
 * Total Submissions: 1.3M
 * Testcase Example:  '[3,2,0,-4]\n1'
 *
 * 给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
 * 
 * 如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos
 * 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。注意：pos
 * 不作为参数进行传递，仅仅是为了标识链表的实际情况。
 * 
 * 不允许修改 链表。
 * 
 * 
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 输入：head = [3,2,0,-4], pos = 1
 * 输出：返回索引为 1 的链表节点
 * 解释：链表中有一个环，其尾部连接到第二个节点。
 * 
 * 
 * 示例 2：
 * 
 * 输入：head = [1,2], pos = 0
 * 输出：返回索引为 0 的链表节点
 * 解释：链表中有一个环，其尾部连接到第一个节点。
 * 
 * 
 * 示例 3：
 * 
 * 输入：head = [1], pos = -1
 * 输出：返回 null
 * 解释：链表中没有环。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 链表中节点的数目范围在范围 [0, 10^4] 内
 * -10^5 <= Node.val <= 10^5
 * pos 的值为 -1 或者链表中的一个有效索引
 * 
 * 
 * 
 * 
 * 进阶：你是否可以使用 O(1) 空间解决此题？
 * 
 */

// @lc code=start
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    // 1.hash，将遍历过的结点保存下来，如果碰到相同的，则返回
    ListNode *detectCycle_hash(ListNode *head) {
        std::unordered_set<ListNode*> visited;
        ListNode *node_ptr = head;
        while(node_ptr) {
            if (visited.count(node_ptr)) {
                return node_ptr;
            }
            visited.insert(node_ptr);

            node_ptr = node_ptr->next;
        }
        
        return nullptr;
    }

    // 2.快慢指针
    // 从相遇点到入环点的距离加上 n-1 圈的环长，恰好等于从链表头部到入环点的距离。
    // 因此，当发现 slow 与 fast 相遇时，我们再额外使用一个指针 ptr。
    // 起始，它指向链表头部；随后，它和 slow 每次向后移动一个位置。
    // 最终，它们会在入环点相遇。
    ListNode *detectCycle(ListNode *head) {
        ListNode* fast_ptr = head;
        ListNode* slow_ptr = head;

        while (fast_ptr != nullptr) {
            if (fast_ptr->next == nullptr) {
                return nullptr;
            }

            fast_ptr = fast_ptr->next->next;
            slow_ptr = slow_ptr->next;

            if (fast_ptr == slow_ptr) {
                ListNode* ptr = head;

                while (ptr != slow_ptr) {
                    ptr = ptr->next;
                    slow_ptr = slow_ptr->next;
                }

                return ptr;
            }

        }
        
        return nullptr;
    }
};
// @lc code=end

