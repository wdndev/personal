# 07 链表

# 1.相交链表

[160. 相交链表 - 力扣（LeetCode）](https://leetcode.cn/problems/intersection-of-two-linked-lists/description/?envType=study-plan-v2\&envId=top-100-liked "160. 相交链表 - 力扣（LeetCode）")

```bash
给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。
```

首先遍历链表 headA，并将链表 headA中的每个节点加入哈希集合中。然后遍历链表 headB，对于遍历到的每个节点，判断该节点是否在哈希集合中：

-   如果当前节点不在哈希集合中，则继续遍历下一个节点；
-   如果当前节点在哈希集合中，则后面的节点都在哈希集合中，即从当前节点开始的所有节点都在两个链表的相交部分，因此在链表 headB 中遍历到的第一个在哈希集合中的节点就是两个链表相交的节点，返回该节点。

如果链表 headB 中的所有节点都不在哈希集合中，则两个链表不相交，返回 null

```c++
class Solution {
public:
    // 哈希表
    // 首先遍历链表 headA，并将链表 headA中的每个节点加入哈希集合中。
    // 然后遍历链表 headBB，对于遍历到的每个节点，判断该节点是否在哈希集合中.
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if (headA == nullptr || headB == nullptr) {
            return nullptr;
        }

        // 无序集合 
        std::unordered_set<ListNode*> visited;
        // 遍历链表A，插入哈希表中
        ListNode* tmp_ptr = headA;
        while (tmp_ptr != nullptr) {
            visited.insert(tmp_ptr);
            tmp_ptr = tmp_ptr->next;
        }

        // 遍历链表B，检查是否出现在hash表中
        tmp_ptr = headB;
        while (tmp_ptr != nullptr) {
            if (visited.count(tmp_ptr)) {
                return tmp_ptr;
            }
            tmp_ptr = tmp_ptr->next;
        }

        return nullptr;
    }
};
```

# 2.反转链表

[206. 反转链表 - 力扣（LeetCode）](https://leetcode.cn/problems/reverse-linked-list/description/?envType=study-plan-v2\&envId=top-100-liked "206. 反转链表 - 力扣（LeetCode）")

```c++
给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。

```

头插法反转链表

```c++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        // 增加头结点，方便插入
        ListNode* head_point = new ListNode();
        head_point->next = nullptr;
        ListNode* node_ptr = head;
        while (node_ptr != nullptr) {
            // 暂存下一个结点
            ListNode* next_node = node_ptr->next;
            // step1:先将插入结点指针修改为头节点的后继
            node_ptr->next = head_point->next;
            // step2： 再将头节点L的后继更新为新插入结点
            head_point->next = node_ptr;

            // 更新下一个节点
            node_ptr = next_node;
        }
        // 注意，增加了头节点，需要去掉
        return head_point->next;
    }
};
```

# 3.回文链表

[234. 回文链表 - 力扣（LeetCode）](https://leetcode.cn/problems/palindrome-linked-list/description/?envType=study-plan-v2\&envId=top-100-liked "234. 回文链表 - 力扣（LeetCode）")

```c++
给你一个单链表的头节点 head ，请你判断该链表是否为回文链表。如果是，返回 true ；否则，返回 false 。

```

1.  将链表中的元素复制到数组中，然后在数组中判断是否回文
2.  快慢指针：将链表的后半部分反转（修改链表结构），然后将前半部分和后半部分进行比较。比较完成后我们应该将链表恢复原样

```c++
class Solution {
public:
    // 1.将链表中的元素复制到数组中，然后再数组中判断是否是回文
    bool isPalindrome(ListNode* head) {
        std::vector<int> arr;
        // 将链表元素复制到数组中
        while (head != nullptr) {
            arr.push_back(head->val);
            head = head->next;
        }
        // 在数组中判断回文
        for (int i = 0, j = arr.size() - 1; i < j; i++, j--) {
            if (arr[i] != arr[j]) {
                return false;
            }
        }

        return true;
    }
};
```

# 4.环形链表

[141. 环形链表 - 力扣（LeetCode）](https://leetcode.cn/problems/linked-list-cycle/description/?envType=study-plan-v2\&envId=top-100-liked "141. 环形链表 - 力扣（LeetCode）")

```c++
给你一个链表的头节点 head ，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递 。仅仅是为了标识链表的实际情况。

如果链表中存在环 ，则返回 true 。 否则，返回 false 。
```

1.  哈希表
2.  快慢指针：慢指针每次只移动一步，而快指针每次移动两步。初始时，慢指针在位置 head，而快指针在位置 head.next。这样一来，如果在移动的过程中，快指针反过来追上慢指针，就说明该链表为环形链表

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

# 5.环形链表Ⅱ

[142. 环形链表 II - 力扣（LeetCode）](https://leetcode.cn/problems/linked-list-cycle-ii/description/?envType=study-plan-v2\&envId=top-100-liked "142. 环形链表 II - 力扣（LeetCode）")

```c++
给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。
```

1.  哈希表
2.  快慢指针：慢指针每次只移动一步，而快指针每次移动两步。初始时，慢指针在位置 head，而快指针在位置 head.next。这样一来，如果在移动的过程中，快指针反过来追上慢指针，就说明该链表为环形链表

```c++
class Solution {
public:
    // 1.哈希表
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

    // 2.快慢指针
    ListNode *detectCycle(ListNode *head) {
        if (head == nullptr || head->next == nullptr) {
            return nullptr;
        }

        ListNode * fast_ptr = head->next;
        ListNode * slow_ptr = head;

        ListNode * result_node;

        while (fast_ptr != slow_ptr) {
            if (fast_ptr == nullptr || fast_ptr->next == nullptr) {
                return nullptr;
            }
            result_node = slow_ptr;
            fast_ptr = fast_ptr->next->next;
            slow_ptr = slow_ptr->next;
        }

        return result_node;
    }
};
```

# 6.合并两个有序链表

[21. 合并两个有序链表 - 力扣（LeetCode）](https://leetcode.cn/problems/merge-two-sorted-lists/description/?envType=study-plan-v2\&envId=top-100-liked "21. 合并两个有序链表 - 力扣（LeetCode）")

```c++
将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

```

```c++
class Solution {
public:
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
```

# 7.两数相加

[2. 两数相加 - 力扣（LeetCode）](https://leetcode.cn/problems/add-two-numbers/description/?envType=study-plan-v2\&envId=top-100-liked "2. 两数相加 - 力扣（LeetCode）")

```c++
给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.

```

同时遍历两个链表，逐位计算它们的和，并与当前位置的进位值相加。

如果两个链表的长度不同，则可以认为长度短的链表的后面有若干个 000 。

此外，如果链表遍历结束后，有 carry>0，还需要在答案链表的后面附加一个节点，节点的值为 carry。

```c++
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
```

# 8.删除链表的倒数第N个结点

[19. 删除链表的倒数第 N 个结点 - 力扣（LeetCode）](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/description/?envType=study-plan-v2\&envId=top-100-liked "19. 删除链表的倒数第 N 个结点 - 力扣（LeetCode）")

```c++
给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
示例 2：

输入：head = [1], n = 1
输出：[]

```

-   整体思路是让前面的指针先移动 `n` 步，之后前后指针共同移动直到前面的指针到尾部为止

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

        // ListNode* tmp_node = slow_ptr->next;
        slow_ptr->next = slow_ptr->next->next;

        return head_node->next;
    }
};
```

# 9.两两交换链表中的结点

[24. 两两交换链表中的节点 - 力扣（LeetCode）](https://leetcode.cn/problems/swap-nodes-in-pairs/description/?envType=study-plan-v2\&envId=top-100-liked "24. 两两交换链表中的节点 - 力扣（LeetCode）")

```python
给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

输入：head = [1,2,3,4]
输出：[2,1,4,3]

```

两两交换

```c++
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        ListNode* head_ptr = head;

        while (head_ptr != nullptr && head_ptr->next != nullptr) {
            int tmp = head_ptr->val;
            head_ptr->val = head_ptr->next->val;
            head_ptr->next->val = tmp;

            head_ptr = head_ptr->next->next;
        }

        return head;
    }
};
```

# 10.k个一组翻转链表

[25. K 个一组翻转链表 - 力扣（LeetCode）](https://leetcode.cn/problems/reverse-nodes-in-k-group/description/?envType=study-plan-v2\&envId=top-100-liked "25. K 个一组翻转链表 - 力扣（LeetCode）")

```python
给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。

k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

输入：head = [1,2,3,4,5], k = 3
输出：[3,2,1,4,5]

```

需要把链表节点按照 k 个一组分组，所以可以使用一个指针 head 依次指向每组的头节点。这个指针每次向前移动 k 步，直至链表结尾。对于每个分组，我们先判断它的长度是否大于等于 k。若是，我们就翻转这部分链表，否则不需要翻转。

```c++
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
```

# 11.随机链表复制

[138. 随机链表的复制 - 力扣（LeetCode）](https://leetcode.cn/problems/copy-list-with-random-pointer/description/?envType=study-plan-v2\&envId=top-100-liked "138. 随机链表的复制 - 力扣（LeetCode）")

```python
给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。

构造这个链表的 深拷贝。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 next 指针和 random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针都不应指向原链表中的节点 。

例如，如果原链表中有 X 和 Y 两个节点，其中 X.random --> Y 。那么在复制链表中对应的两个节点 x 和 y ，同样有 x.random --> y 。

返回复制链表的头节点。

用一个由 n 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 [val, random_index] 表示：

val：一个表示 Node.val 的整数。
random_index：随机指针指向的节点索引（范围从 0 到 n-1）；如果不指向任何节点，则为  null 。
你的代码 只 接受原链表的头节点 head 作为传入参数。
```

我们用哈希表记录每一个节点对应新节点的创建情况。遍历该链表的过程中，我们检查「当前节点的后继节点」和「当前节点的随机指针指向的节点」的创建情况。如果这两个节点中的任何一个节点的新节点没有被创建，我们都立刻递归地进行创建。当我们拷贝完成，回溯到当前层时，我们即可完成当前节点的指针赋值。注意一个节点可能被多个其他节点指向，因此我们可能递归地多次尝试拷贝某个节点，为了防止重复拷贝，我们需要首先检查当前节点是否被拷贝过，如果已经拷贝过，我们可以直接从哈希表中取出拷贝后的节点的指针并返回即可。

```c++
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if (head == nullptr) {
            return nullptr;
        }

        if (!m_cache_node.count(head)) {
            Node* head_new = new Node(head->val);
            m_cache_node[head] = head_new;
            head_new->next = this->copyRandomList(head->next);
            head_new->random = this->copyRandomList(head->random);
        }

        return m_cache_node[head];
    }
private:
    std::unordered_map<Node*, Node*> m_cache_node;
};
```

# 12.排序链表

[148. 排序链表 - 力扣（LeetCode）](https://leetcode.cn/problems/sort-list/description/?envType=study-plan-v2\&envId=top-100-liked "148. 排序链表 - 力扣（LeetCode）")

```python
给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表 。


```

冒泡排序法：超时

-   不断从第一个结点往表尾扫描；
-   如果相邻前驱与后继结点存在逆序，就将前驱和后继结点数据交换，直到一趟链表中不存在逆序。

```c++
class Solution {
public:
    ListNode* sortList(ListNode* head) {
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
};
```

投机取巧，遍历一次链表，把值丢入自排序的数组，最后同时遍历集合和链表进行值覆盖。

```c++
class Solution {
public:
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
```

# 13.合并k个升序链表

[23. 合并 K 个升序链表 - 力扣（LeetCode）](https://leetcode.cn/problems/merge-k-sorted-lists/description/?envType=study-plan-v2\&envId=top-100-liked "23. 合并 K 个升序链表 - 力扣（LeetCode）")

```python
给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

 

示例 1：

输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6
```

我们可以想到一种最朴素的方法：用一个变量 ans 来维护以及合并的链表，第 iii 次循环把第 i 个链表和 ans 合并，答案保存到 ans 中。

```c++
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
```

# 14.LRU缓存

[146. LRU 缓存 - 力扣（LeetCode）](https://leetcode.cn/problems/lru-cache/description/?envType=study-plan-v2\&envId=top-100-liked "146. LRU 缓存 - 力扣（LeetCode）")

```c++
请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
实现 LRUCache 类：

- LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
- int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
- void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存

在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。
函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。
```

```c++
class LRUCache {
public:
    using Pair = std::pair<int, int>;
    using List = std::list<Pair>;
    using Map = std::unordered_map<int, typename List::iterator>;

    LRUCache(int capacity) {
        m_capacity = capacity;
    }
    
    int get(int key) {
        // 查找key是否在哈希表中
        typename Map::iterator map_itor = m_map.find(key);
        // 不存在，返回-1
        if (map_itor == m_map.end())
            return -1;

        // 如果存在
        // 1.链表要将数据删除
        // 2.在将数据加入到链表队头
        // 目的是为了维护链表队头的hi最近访问的数据

        // 取出哈希表的value值，也就是链表节点
        typename List::iterator list_itor = map_itor->second;

        // 创建新的键值对
        std::pair<int, int> list_pair = std::make_pair(list_itor->first, list_itor->second);

        // 从链表中删除该节点
        m_list.erase(list_itor);

        // 将数据加入到队头
        m_list.push_front(list_pair);

        // 更新哈希表
        m_map[key] = m_list.begin();

        return list_pair.second;
    }
    
    void put(int key, int value) {
        // 查找key是否在哈希表中
        typename Map::iterator itor = m_map.find(key);
        // 如果存在，则要将老数据从哈希表和链表中移除
        if (itor != m_map.end())
        {
            m_list.erase(itor->second);
            m_map.erase(itor);
        }

        // 插入到链表头
        m_list.push_front(std::make_pair(key, value));
        // 将链表头放入hash表中
        m_map[key] = m_list.begin();

        // 当链表大小超过阈值后，删除
        if (m_list.size() > m_capacity)
        {
            int end_key = m_list.back().first;
            m_list.pop_back();
            m_map.erase(end_key);
        }
    }
private:
    int m_capacity;
    List m_list;
    Map m_map;
};

```
