# 3.链表双指针

## 1.双指针简介

> **双指针（Two Pointers）**：指的是在遍历元素的过程中，不是使用单个指针进行访问，而是使用两个指针进行访问，从而达到相应的目的。如果两个指针方向相反，则称为「对撞时针」。如果两个指针方向相同，则称为「快慢指针」。如果两个指针分别属于不同的数组 / 链表，则称为「分离双指针」。

而在单链表中，因为遍历节点只能顺着 `next` 指针方向进行，所以对于链表而言，一般只会用到「快慢指针」和「分离双指针」。其中链表的「快慢指针」又分为「起点不一致的快慢指针」和「步长不一致的快慢指针」。这几种类型的双指针所解决的问题也各不相同，下面我们一一进行讲解。

## 2.起点不一致的快慢指针

> **起点不一致的快慢指针**：指的是两个指针从同一侧开始遍历链表，但是两个指针的起点不一样。 快指针 `fast` 比慢指针 `slow` 先走 `n` 步，直到快指针移动到链表尾端时为止。

求解步骤：

1.  使用两个指针 `slow`、`fast`。`slow`、`fast` 都指向链表的头节点，即：`slow = head`，`fast = head`。
2.  先将快指针向右移动 `n` 步。然后再同时向右移动快、慢指针。
3.  等到快指针移动到链表尾部（即 `fast == None`）时跳出循环体。

代码模板：

```python
slow = head
fast = head

while n:
    fast = fast.next
    n -= 1
while fast:
    fast = fast.next
    slow = slow.next

```

适用范围：

起点不一致的快慢指针主要用于**找到链表中倒数第 k 个节点、删除链表倒数第 N 个节点**等。

## 3.步长不一致的快慢指针

> **步长不一致的快慢指针**：指的是两个指针从同一侧开始遍历链表，两个指针的起点一样，但是步长不一致。例如，慢指针 `slow` 每次走 `1` 步，快指针 `fast` 每次走两步。直到快指针移动到链表尾端时为止。

求解步骤：

1.  使用两个指针 `slow`、`fast`。`slow`、`fast` 都指向链表的头节点。
2.  在循环体中将快、慢指针同时向右移动，但是快、慢指针的移动步长不一致。比如将慢指针每次移动 `1` 步，即 `slow = slow.next`。快指针每次移动 `2` 步，即 `fast = fast.next.next`。
3.  等到快指针移动到链表尾部（即 `fast == None`）时跳出循环体。

代码模板：

```python
fast = head
slow = head

while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
```

适用范围：

步长不一致的快慢指针适合**寻找链表的中点、判断和检测链表是否有环、找到两个链表的交点**等问题。

## 4.分离双指针

> **分离双指针**：两个指针分别属于不同的链表，两个指针分别在两个链表中移动。

求解步骤：

1.  使用两个指针 `left_1`、`left_2`。`left_1` 指向第一个链表头节点，即：`left_1 = list1`，`left_2` 指向第二个链表头节点，即：`left_2 = list2`。
2.  当满足一定条件时，两个指针同时右移，即 `left_1 = left_1.next`、`left_2 = left_2.next`。
3.  当满足另外一定条件时，将 `left_1` 指针右移，即 `left_1 = left_1.next`。
4.  当满足其他一定条件时，将 `left_2` 指针右移，即 `left_2 = left_2.next`。
5.  当其中一个链表遍历完时或者满足其他特殊条件时跳出循环体。

代码模板：

```python
left_1 = list1
left_2 = list2

while left_1 and left_2:
    if 一定条件 1:
        left_1 = left_1.next
        left_2 = left_2.next
    elif 一定条件 2:
        left_1 = left_1.next
    elif 一定条件 3:
        left_2 = left_2.next

```

适用范围：

分离双指针一般用于**有序链表合并**等问题。

## 5.实战题目

### 5.1 删除链表的倒数第N个结点

[19. 删除链表的倒数第 N 个结点 - 力扣（LeetCode）](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/ "19. 删除链表的倒数第 N 个结点 - 力扣（LeetCode）")

```c++
给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]

```

起点不一致的快慢指针

```c++
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        // 增加头节点
        ListNode* head_node = new ListNode();
        head_node->next = head;

        // 快慢指针
        ListNode* fast_ptr = head_node->next;
        ListNode* slow_ptr = head_node;

        // 快指针先移动n步
        for (int i = 0; i < n; i++) {
            fast_ptr = fast_ptr->next;
        }

        // 快慢指针再一起移动
        while (fast_ptr != nullptr) {
            fast_ptr = fast_ptr->next;
            slow_ptr = slow_ptr->next;
        }

        ListNode* tmp_node = slow_ptr->next;
        slow_ptr->next = slow_ptr->next->next;
        delete tmp_node;

        return head_node->next;
    }
};
```

### 5.2 链表的中间结点

[876. 链表的中间结点 - 力扣（LeetCode）](https://leetcode.cn/problems/middle-of-the-linked-list/ "876. 链表的中间结点 - 力扣（LeetCode）")

```c++
给你单链表的头结点 head ，请你找出并返回链表的中间结点。

如果有两个中间结点，则返回第二个中间结点。

输入：head = [1,2,3,4,5]
输出：[3,4,5]
解释：链表只有一个中间结点，值为 3 。

```

1、单指针：先找到链表的长度，再遍历一次找到中间结点

2、双指针：步长不一致的快慢指针

```c++
class Solution {
public:
    ListNode* middleNode(ListNode* head) {
        ListNode* fast_ptr = head;
        ListNode* slow_ptr = head;

        while (fast_ptr != nullptr && fast_ptr->next != nullptr) {
            fast_ptr = fast_ptr->next->next;
            slow_ptr = slow_ptr->next;
        }

        return slow_ptr;
    }
};
```

### 5.3环形链表

[141. 环形链表 - 力扣（LeetCode）](https://leetcode.cn/problems/linked-list-cycle/description/ "141. 环形链表 - 力扣（LeetCode）")

```c++
给你一个链表的头节点 head ，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递 。仅仅是为了标识链表的实际情况。

如果链表中存在环 ，则返回 true 。 否则，返回 false 。

输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点。

```

1、哈希表：遍历所有节点，每次遍历节点之前，使用哈希表判断该节点是否被访问过

2、快慢指针：步长不一致的快慢指针

```c++
class Solution {
public:
    // 1.哈希表
    bool hasCycle(ListNode *head) {
        std::unordered_set<ListNode *> visited;

        while (head != nullptr) {
            if (visited.count(head)) {
                return true;
            }

            visited.insert(head);
            head = head->next;
        }
        return false;
    }
    // 2.快慢指针
    bool hasCycle2(ListNode *head) {
        if (head == nullptr || head->next == nullptr) {
            return false;
        }

        ListNode * fast_ptr = head->next;
        ListNode * slow_ptr = head;

        while (fast_ptr != slow_ptr) {
            if (fast_ptr == nullptr || fast_ptr->next == nullptr) {
                return false;
            }

            fast_ptr = fast_ptr->next->next;
            slow_ptr = slow_ptr->next;
        }

        return true;
    }
};
```

### 5.4 环形链表Ⅱ

[142. 环形链表 II - 力扣（LeetCode）](https://leetcode.cn/problems/linked-list-cycle-ii/description/ "142. 环形链表 II - 力扣（LeetCode）")

```c++
给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。

不允许修改 链表。

输入：head = [3,2,0,-4], pos = 1
输出：返回索引为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。

```

1、哈希表

2、快慢指针

1.  利用两个指针，一个慢指针 `slow` 每次前进一步，快指针 `fast` 每次前进两步（两步或多步效果是等价的）。
2.  如果两个指针在链表头节点以外的某一节点相遇（即相等）了，那么说明链表有环。
3.  否则，如果（快指针）到达了某个没有后继指针的节点时，那么说明没环。
4.  如果有环，则再定义一个指针 `ans`，和慢指针一起每次移动一步，两个指针相遇的位置即为入口节点。

```c++
class Solution {
public:
    // 1、哈希表
    ListNode *detectCycle1(ListNode *head) {
        std::unordered_set<ListNode *> visited;

        while (head != nullptr) {
            if (visited.count(head)) {
                return head;
            }

            visited.insert(head);
            head = head->next;
        }
        return nullptr;
    }
    // 2、快慢指针
    ListNode *detectCycle(ListNode *head) {
        ListNode *slow = head, *fast = head;
        while (fast != nullptr) {
            slow = slow->next;
            if (fast->next == nullptr) {
                return nullptr;
            }
            fast = fast->next->next;
            if (fast == slow) {
                ListNode *ptr = head;
                while (ptr != slow) {
                    ptr = ptr->next;
                    slow = slow->next;
                }
                return ptr;
            }
        }
        return nullptr;
    }
};
```

### 5.5 合并两个有序链表

[21. 合并两个有序链表 - 力扣（LeetCode）](https://leetcode.cn/problems/merge-two-sorted-lists/description/ "21. 合并两个有序链表 - 力扣（LeetCode）")

```c++
将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 


输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]

```

分离双指针

1.  使用哑节点 `dummy_head` 构造一个头节点，并使用 `curr` 指向 `dummy_head` 用于遍历。
2.  然后判断 `list1` 和 `list2` 头节点的值，将较小的头节点加入到合并后的链表中。并向后移动该链表的头节点指针。
3.  然后重复上一步操作，直到两个链表中出现链表为空的情况。
4.  将剩余链表链接到合并后的链表中。
5.  将哑节点 `dummy_dead` 的下一个链节点 `dummy_head.next` 作为合并后有序链表的头节点返回。

```c++
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
```
