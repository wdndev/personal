# 08 Tree

# 1.二叉树的中序遍历

[94. 二叉树的中序遍历 - 力扣（LeetCode）](https://leetcode.cn/problems/binary-tree-inorder-traversal/description/?envType=study-plan-v2\&envId=top-100-liked "94. 二叉树的中序遍历 - 力扣（LeetCode）")

```python
给定一个二叉树的根节点 root ，返回 它的 中序 遍历 。
```

1.  递归
2.  迭代：**借助栈**

```c++
class Solution {
public:
    // 1.递归
    vector<int> inorderTraversal1(TreeNode* root) {
        std::vector<int> ans;
        this->inorder(root, ans);
        return ans;
    }

    void inorder(TreeNode* root, std::vector<int>& ans) {
        if (root == nullptr) {
            return;
        }
        this->inorder(root->left, ans);
        ans.push_back(root->val);
        this->inorder(root->right, ans);
    }

    // 2.迭代
    vector<int> inorderTraversal(TreeNode* root) {
        std::vector<int> ans;
        std::stack<TreeNode*> stk;

        while (root != nullptr || !stk.empty()) {
            while (root != nullptr) {
                stk.push(root);
                root = root->left;
            }
            root = stk.top();
            stk.pop();
            ans.push_back(root->val);
            root = root->right;
        }
        return ans;
    }
};
```

# 2.二叉树的最大深度

[104. 二叉树的最大深度 - 力扣（LeetCode）](https://leetcode.cn/problems/maximum-depth-of-binary-tree/description/?envType=study-plan-v2\&envId=top-100-liked "104. 二叉树的最大深度 - 力扣（LeetCode）")

```python
给定一个二叉树 root ，返回其最大深度。

二叉树的 最大深度 是指从根节点到最远叶子节点的最长路径上的节点数。
```

1.  递归，深度优先搜索
2.  迭代，广度优先搜索，借助队列，

```c++
class Solution {
public:
    // 1.递归, 深度优先搜索
    int maxDepth1(TreeNode* root) {
        if (root == nullptr) {
            return 0;
        }
        return std::max(this->maxDepth(root->left), this->maxDepth(root->right)) + 1;
    }
    
    // 2.迭代，广度优先搜索
    int maxDepth(TreeNode* root) {
        if (root == nullptr) {
            return 0;
        }

        std::queue<TreeNode*> que;
        que.push(root);
        int count_num = 0;

        while (!que.empty()) {
            int size = que.size();
            while (size > 0) {
                TreeNode* tmp_node = que.front();
                que.pop();

                if (tmp_node->left != nullptr) {
                    que.push(tmp_node->left);
                }
                if (tmp_node->right != nullptr) {
                    que.push(tmp_node->right);
                }

                size -= 1;
            }
            count_num += 1;
        }

        return count_num;
    }
};
```

# 3.翻转二叉树

[226. 翻转二叉树 - 力扣（LeetCode）](https://leetcode.cn/problems/invert-binary-tree/description/?envType=study-plan-v2\&envId=top-100-liked "226. 翻转二叉树 - 力扣（LeetCode）")

```python
给你一棵二叉树的根节点 root ，翻转这棵二叉树，并返回其根节点。

```

从根节点开始，递归地对树进行遍历，并从叶子节点先开始翻转。如果当前遍历到的节点 root的左右两棵子树都已经翻转，那么我们只需要交换两棵子树的位置，即可完成以 root 为根节点的整棵子树的翻转。

```c++
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (root == nullptr) {
            return nullptr;
        }

        TreeNode* right = this->invertTree(root->right);
        TreeNode* left = this->invertTree(root->left);

        root->left = right;
        root->right = left;

        return root;
    }
};
```

# 4.对称二叉树

[101. 对称二叉树 - 力扣（LeetCode）](https://leetcode.cn/problems/symmetric-tree/description/?envType=study-plan-v2\&envId=top-100-liked "101. 对称二叉树 - 力扣（LeetCode）")

```python
给你一个二叉树的根节点 root ， 检查它是否轴对称。

输入：root = [1,2,2,3,4,4,3]
输出：true

```

通过「同步移动」两个指针的方法来遍历这棵树，p 指针和 q 指针一开始都指向这棵树的根，随后 p 右移时，q 左移，p 左移时，q 右移。每次检查当前 p 和 q 节点的值是否相等，如果相等再判断左右子树是否对称。

```c++
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        return this->is_similar(root->left, root->right);
    }

    bool is_similar(TreeNode* tree1, TreeNode* tree2) {
        // 递归终止条件：
        // 1.两个结点为空
        // 2.其中一个为空，一个不为空
        // 3.两个结点的值不相等
        if (tree1 == nullptr && tree2 == nullptr) {
            return true;
        }

        if (tree1 == nullptr || tree2 == nullptr) {
            return false;
        }

        if (tree1->val != tree2->val) {
            return false;
        }

        // 再次递归比较：
        // 1. T1的左孩子和T2的右孩子
        // 2. T1的有孩子和T2的左孩子
        return (this->is_similar(tree1->left, tree2->right) 
                && this->is_similar(tree1->right, tree2->left));
    }
};
```

# 5.二叉树的直径

[543. 二叉树的直径 - 力扣（LeetCode）](https://leetcode.cn/problems/diameter-of-binary-tree/description/?envType=study-plan-v2\&envId=top-100-liked "543. 二叉树的直径 - 力扣（LeetCode）")

```python
给你一棵二叉树的根节点，返回该树的 直径 。

二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点 root 。

两节点之间路径的 长度 由它们之间边数表示。

输入：root = [1,2,3,4,5]
输出：3
解释：3 ，取路径 [4,2,1,3] 或 [5,2,1,3] 的长度。

```

主要计算当前结点的最大直径：当前结点的最大直径 = 左子树深度 + 右子树深度

递归函数返回的为当前结点的最大深度

```c++
class Solution {
public:
    int diameterOfBinaryTree(TreeNode* root) {
        int max_diam = -1;
        this->dfs(root, max_diam);
        
        return max_diam;
    }

    int dfs(TreeNode* root, int& max_diam) {
        if (root == nullptr) {
            return 0;
        }

        int left_diam = this->dfs(root->left, max_diam);
        int right_diam = this->dfs(root->right, max_diam);

        // 每个结点最大直径（左子树深度 + 右子树深度）
        // 更新最大深度
        max_diam = std::max(left_diam + right_diam, max_diam);

        // 返回当前结点的最大深度
        return std::max(left_diam, right_diam) + 1;
    }
};
```

# 6.二叉树的层序遍历

[102. 二叉树的层序遍历 - 力扣（LeetCode）](https://leetcode.cn/problems/binary-tree-level-order-traversal/description/?envType=study-plan-v2\&envId=top-100-liked "102. 二叉树的层序遍历 - 力扣（LeetCode）")

```c++
给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）。

```

1.  广度优先遍历：使用队列遍历
2.  深度优先遍历，递归时使用变量还保存层数

```c++
class Solution {
public:
    // 1.广度优先遍历
    vector<vector<int>> levelOrder1(TreeNode* root) {
        std::vector<std::vector<int>> ans;
        if (root == nullptr) {
            return ans;
        }

        std::queue<TreeNode*> que;
        que.push(root);

        while (!que.empty()) {
            std::vector<int> level_node;
            int node_count = que.size();
            for (int i = 0; i < node_count; i++) 
            {
                TreeNode* tmp_node = que.front();
                que.pop();

                level_node.push_back(tmp_node->val);

                if (tmp_node->left != nullptr) {
                    que.push(tmp_node->left);
                }

                if (tmp_node->right != nullptr) {
                    que.push(tmp_node->right);
                }
            }
            
            ans.push_back(std::move(level_node));
        }

        return ans;
    }

    // 2.深度优先遍历
    vector<vector<int>> levelOrder(TreeNode* root) {
        std::vector<std::vector<int>> ans;
        if (root == nullptr) {
            return ans;
        }

        this->dfs(root, 0, ans);

        return ans;
    }

    void dfs(TreeNode* root, int depth, std::vector<std::vector<int>>& ans) {
        if (root == nullptr) {
            return;
        }

        if (depth >= ans.size()) {
            ans.push_back(std::vector<int>{});
        }
        ans[depth].push_back(root->val);

        this->dfs(root->left, depth + 1, ans);
        this->dfs(root->right, depth + 1, ans);
    }
};
```

# 7.将有序数组转换为二叉树

[108. 将有序数组转换为二叉搜索树 - 力扣（LeetCode）](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/description/?envType=study-plan-v2\&envId=top-100-liked "108. 将有序数组转换为二叉搜索树 - 力扣（LeetCode）")

```python
给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。

高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。

输入：nums = [-10,-3,0,5,9]
输出：[0,-3,9,-10,null,5]
解释：[0,-10,5,null,-3,null,9] 也将被视为正确答案：

```

中序遍历，使用二分查找的思想，开始构造左右子树

```c++
class Solution {
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return this->dfs(nums, 0, nums.size() - 1);
    }

    TreeNode* dfs(std::vector<int>& nums, int left, int right) {
        if (left > right) {
            return nullptr;
        }

        // 总是选择中间的结点作为根节点
        int mid = left + (right - left) / 2;
        // int mid = (left + right) / 2;

        TreeNode* root = new TreeNode(nums[mid]);
        root->left = this->dfs(nums, left, mid - 1);
        root->right = this->dfs(nums, mid + 1, right);

        return root;
    }
};
```

# 8.验证二叉搜索树

[98. 验证二叉搜索树 - 力扣（LeetCode）](https://leetcode.cn/problems/validate-binary-search-tree/description/?envType=study-plan-v2\&envId=top-100-liked "98. 验证二叉搜索树 - 力扣（LeetCode）")

```python
给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。

有效 二叉搜索树定义如下：

节点的左子树只包含 小于 当前节点的数。
节点的右子树只包含 大于 当前节点的数。
所有左子树和右子树自身必须也是二叉搜索树。
```

BST --> 中序遍历是递增的

```c++
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        if (root == nullptr) {
            return true;
        }

        // 遍历左子树
        if (!this->isValidBST(root->left)) {
            return false;
        }

        // 当前结点不大于父节点，不是排序二叉树
        if (root->val <= m_father_value) {
            return false;
        } else {
            // 记录父节点
            m_father_value = root->val;
        }

        // 遍历右子树
        if (!this->isValidBST(root->right)) {
            return false;
        }

        // 子树遍历完成
        return true;
    }

private:
    // 用于存储最新遍历的父亲结点值
    long m_father_value = LONG_MIN;
};
```

# 9.二叉搜索树中的第k小的元素

[230. 二叉搜索树中第K小的元素 - 力扣（LeetCode）](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/description/?envType=study-plan-v2\&envId=top-100-liked "230. 二叉搜索树中第K小的元素 - 力扣（LeetCode）")

```json
给定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 个最小元素（从 1 开始计数）。


```

1.  中序遍历+数组 ：中序遍历先存在数组中，然后再将数组中的第k个元素取出
2.  中序遍历 + 索引 ： 中序遍历到第k个值返回

```c++
class Solution {
public:
    // 中序遍历
    int kthSmallest(TreeNode* root, int k) {
        std::vector<int> ans;
        
        int val = -1;
        this->inorder(root, k, val);

        return val;
    }

    void inorder(TreeNode* root, int idx, int& val) {
        if (root == nullptr) {
            return;
        }
        
        this->inorder(root->left, idx, val);
        
        m_idx++;
        if (m_idx == idx) {
            val = root->val;
            return;
        }
        // idx--;
        
        this->inorder(root->right, idx, val);
    }

    // void inorder(TreeNode* root, std::vector<int>& ans) {
    //     if (root == nullptr) {
    //         return;
    //     }
    //     this->inorder(root->left, ans);
    //     ans.push_back(root->val);
    //     this->inorder(root->right, ans);
    // }
    int m_idx = 0;
};
```

# 10.二叉树的右视图

[199. 二叉树的右视图 - 力扣（LeetCode）](https://leetcode.cn/problems/binary-tree-right-side-view/description/?envType=study-plan-v2\&envId=top-100-liked "199. 二叉树的右视图 - 力扣（LeetCode）")

```json
给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。


输入: [1,2,3,null,5,null,4]
输出: [1,3,4]

```

层序遍历，每一层的最后一个结点

```c++
class Solution {
public:
    // 层序遍历，每一层的最后一个
    vector<int> rightSideView(TreeNode* root) {
        std::vector<int> ans;
        if (root == nullptr) {
            return ans;
        }

        std::queue<TreeNode*> que;
        que.push(root);

        while(!que.empty()) {
            std::vector<int> level_node;
            int level_size = que.size();
            for (int i = 0; i < level_size; i++) {
                TreeNode* tmp_node = que.front();
                que.pop();

                level_node.push_back(tmp_node->val);

                if (tmp_node->left != nullptr) {
                    que.push(tmp_node->left);
                }

                if (tmp_node->right != nullptr) {
                    que.push(tmp_node->right);
                }
            }
            ans.push_back(level_node[level_size - 1]);
        }

        return ans;
    }
};
```

# 11.二叉树展开为链表

[114. 二叉树展开为链表 - 力扣（LeetCode）](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/description/?envType=study-plan-v2\&envId=top-100-liked "114. 二叉树展开为链表 - 力扣（LeetCode）")

```json
给你二叉树的根结点 root ，请你将它展开为一个单链表：

展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
展开后的单链表应该与二叉树 先序遍历 顺序相同。

输入：root = [1,2,5,3,4,null,6]
输出：[1,null,2,null,3,null,4,null,5,null,6]

```

1.  先序遍历 + 结果连接
2.  反先序遍历 ： 将前序遍历反过来遍历，那么第一次访问的就是前序遍历中最后一个节点。那么可以调整最后一个节点，再将最后一个节点保存到pre里，再调整倒数第二个节点，将它的右子树设置为pre，再调整倒数第三个节点，依次类推直到调整完毕。和反转链表的递归思路是一样的。

```c++
class Solution {
public:
    // 1. 先序遍历 + 结构连接
    void flatten1(TreeNode* root) {
        std::vector<TreeNode*> node_vec;
        this->pre_order(root, node_vec);

        TreeNode* pre_node;
        TreeNode* curr_node;
        for (int i = 1; i < node_vec.size(); i++) {
            pre_node = node_vec[i - 1];
            curr_node = node_vec[i];

            pre_node->right = curr_node;
            pre_node->left = nullptr;
        }
    }

    void pre_order(TreeNode* root, std::vector<TreeNode*>& node_vec) {
        if (root == nullptr) {
            return;
        }
        node_vec.push_back(root);
        this->pre_order(root->left, node_vec);
        this->pre_order(root->right, node_vec);
    }

    // 2. 反先序遍历
    TreeNode* pre_node;
    void flatten(TreeNode* root) {
        if (root == nullptr) {
            return;
        }

        // 右 -> 左 -> 中
        this->flatten(root->right);
        this->flatten(root->left);
        
        root->left = nullptr;
        root->right = pre_node;
        pre_node = root;
    }
};
```

# 12.从前序和中序遍历中构造二叉树

[105. 从前序与中序遍历序列构造二叉树 - 力扣（LeetCode）](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/?envType=study-plan-v2\&envId=top-100-liked "105. 从前序与中序遍历序列构造二叉树 - 力扣（LeetCode）")

```json
给定两个整数数组 preorder 和 inorder ，其中 preorder 是二叉树的先序遍历， inorder 是同一棵树的中序遍历，请构造二叉树并返回其根节点。


输入: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
输出: [3,9,20,null,null,15,7]

```

int length = rootIndex - inStartIndex;
一颗树的先序序列和中序序列能确定这棵树。

-   先序序列：根，左，右
-   中序序列：左，根，右
-   后序序列：左，右，根

表示子树不存在：

-   序列长度为零，代表子树不存在
-   给定起始位置索引和终止位置索引。起始位置和终止位置不合法，也就是终止位置大于起始位置来代表序列不存在，即代表字数不存在

```c++
class Solution {
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        return this->pre_inorder_build_tree(preorder, 0, preorder.size() - 1, inorder, 0, inorder.size() - 1);
    }

private:
    TreeNode* pre_inorder_build_tree(std::vector<int>& preorder, int pre_start_idx, int pre_end_idx,
                                    std::vector<int>& inorder, int in_start_idx, int in_end_idx) {
        if (pre_start_idx > pre_end_idx) {
            return nullptr;
        }

        // 创建根节点，根节点的值使用前序遍历的第一个
        TreeNode* root = new TreeNode(preorder[pre_start_idx]);

        // 在中序遍历中找到根节点，划分为两个数组，分别是左右子树的，
        int root_idx = in_start_idx;
        for (; root_idx <= in_end_idx; root_idx++) {
            if (root->val == inorder[root_idx]) {
                break;
            }
        }

        // 左子树的长度
        int left_lens = root_idx - in_start_idx;

        // 创建左子树
        root->left = this->pre_inorder_build_tree(preorder, pre_start_idx + 1, pre_start_idx + left_lens, 
                                                  inorder, in_start_idx, root_idx - 1);
        // 创建右子树
        root->right = this->pre_inorder_build_tree(preorder, pre_start_idx + left_lens + 1, pre_end_idx, 
                                                  inorder, root_idx + 1, in_end_idx);

        return root;
    }
};
```

# 13.路径总和Ⅲ

[437. 路径总和 III - 力扣（LeetCode）](https://leetcode.cn/problems/path-sum-iii/description/?envType=study-plan-v2\&envId=top-100-liked "437. 路径总和 III - 力扣（LeetCode）")

```json
给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。

路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。
```

**深度优先遍历**：访问每一个节点 node，检测以 node为起始节点且向下延深的路径有多少种。我们递归遍历每一个节点的所有可能的路径，然后将这些路径数目加起来即为返回结果。

-   `root_sum `： 从结点root为起点满足路径和为 target\_sum 的路径数目
-   `pathSum `: pathSum

```c++
class Solution {
public:
    // 递归遍历每个结点，求路径总和
    int pathSum(TreeNode* root, int targetSum) {
        if (root == nullptr) {
            return 0;
        }
        
        int ret = this->root_sum(root, targetSum);
        ret += this->pathSum(root->left, targetSum);
        ret += this->pathSum(root->right, targetSum);

        return ret;
    }

    // 从结点root为起点满足路径和为 target_sum 的路径数目
    int root_sum(TreeNode* root, long target_sum) {
        if (root == nullptr) {
            return 0;
        }

        int ret = 0;
        if (root->val == target_sum) {
            ret++;
        }

        // 左右子树分别向下扩展
        ret += this->root_sum(root->left, target_sum - root->val);
        ret += this->root_sum(root->right, target_sum - root->val);

        return ret;
    }
};
```

# 14.二叉树的最近公共祖先

[236. 二叉树的最近公共祖先 - 力扣（LeetCode）](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description/?envType=study-plan-v2\&envId=top-100-liked "236. 二叉树的最近公共祖先 - 力扣（LeetCode）")

```json
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
```

判断条件

1.  左子树存在 p 或 q，右子树存在p或q，则此结点是公共祖先
2.  左子树或右子树其中一方存在p和q，另一方没有，那这个结点就是公共祖先

```c++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        TreeNode* father = nullptr;
        this->dfs(root, p, q, father);

        return father;
    }
private:
    bool dfs(TreeNode* root, TreeNode* p, TreeNode* q, TreeNode*& father) {
        // 到达叶子结点，返回false
        if (root == nullptr) {
            return false;
        }

        // 搜索左右子树是否是公共根节点
        bool left = dfs(root->left, p, q, father);
        bool right = dfs(root->right, p, q, father);

        // 判断条件
        // 1.左子树存在 p 或 q，右子树存在p或q，则此结点是公共祖先
        // 2.左子树或右子树其中一方存在p和q，另一方没有，那这个结点就是公共祖先
        if ((left && right) || ((root->val == p->val || root->val == q->val) && (left || right))) {
            father = root;
        }

        // 公共祖先条件
        return left || right || (root->val == p->val || root->val == q->val);
    }
};
```

# 15.二叉树的最大路径和

[124. 二叉树中的最大路径和 - 力扣（LeetCode）](https://leetcode.cn/problems/binary-tree-maximum-path-sum/description/?envType=study-plan-v2\&envId=top-100-liked "124. 二叉树中的最大路径和 - 力扣（LeetCode）")

```json
二叉树中的 路径 被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。

路径和 是路径中各节点值的总和。

给你一个二叉树的根节点 root ，返回其 最大路径和 。
```

首先，考虑实现一个简化的函数 `dfs(TreeNode* root, int& max_sum)`，该函数计算二叉树中的一个节点的最大贡献值，具体而言，就是在以该节点为根节点的子树中寻找以该节点为起点的一条路径，使得该路径上的节点值之和最大。

具体而言，该函数的计算如下。

1.  空节点的最大贡献值等于 0。
2.  非空节点的最大贡献值等于节点值与其子节点中的最大贡献值之和（对于叶节点而言，最大贡献值等于节点值）。

```c++
class Solution {
public:
    int maxPathSum(TreeNode* root) {
        int max_sum = INT_MIN;
        this->dfs(root, max_sum);
        return max_sum;
    }

    int dfs(TreeNode* root, int& max_sum) {
        if (root == nullptr) {
            return 0;
        }

        // 递归计算左右子树最大结点值，
        // 如果计算出的结点值为负数，则抛弃，变为0
        int left_max = std::max(this->dfs(root->left, max_sum), 0);
        int right_max = std::max(this->dfs(root->right, max_sum), 0);

        // 计算所在结点的最大路径和，并更新max_sum
        int node_max = root->val + left_max + right_max;
        max_sum = std::max(max_sum, node_max);

        // 返回该结点的最大值，方便递归
        return root->val + std::max(left_max, right_max);
    }
};
```
