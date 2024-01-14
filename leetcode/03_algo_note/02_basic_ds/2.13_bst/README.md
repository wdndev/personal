# 13.二叉搜索树

## 1.二叉搜索树简介

**二叉搜索树（Binary Search Tree）**：也叫做二叉查找树、有序二叉树或者排序二叉树。是指一棵空树或者具有下列性质的二叉树：

-   如果任意节点的左子树不为空，则左子树上所有节点的值均小于它的根节点的值。
-   如果任意节点的右子树不为空，则右子树上所有节点的值均大于它的根节点的值。
-   任意节点的左子树、右子树均为二叉搜索树。

二叉树具有一个特性，即：**左子树的节点值 < 根节点值 < 右子树的节点值**。

根据这个特性，**如果以中序遍历的方式遍历整个二叉搜索树时，会得到一个递增序列**。

## 2.二叉搜索树操作

### 2.1 查找

> **二叉搜索树的查找**：在二叉搜索树中查找值为 `val` 的节点。

按照二叉搜索树的定义，在进行元素查找时，只需要根据情况判断需要往左还是往右走。这样，每次根据情况判断都会缩小查找范围，从而提高查找效率。二叉树的查找步骤如下：

1.  如果二叉搜索树为空，则查找失败，结束查找，并返回空指针节点 `None`。
2.  如果二叉搜索树不为空，则将要查找的值 `val` 与二叉搜索树根节点的值 `root.val` 进行比较：
    1.  如果 `val == root.val`，则查找成功，结束查找，返回被查找到的节点。
    2.  如果 `val < root.val`，则递归查找左子树。
    3.  如果 `val > root.val`，则递归查找右子树。

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None) -> None:
        self.val = val
        self.left = left
        self.right = right

class BST:
    def search_bst(self, root:TreeNode, val:int) -> TreeNode:
        """ 二叉搜索树查找
        """
        if not root:
            return None
        
        if val == root.val:
            return root
        elif val < root.val:
            return self.search_bst(root.left, val)
        else:
            return self.search_bst(root.right, val)
```

-   二叉搜索树的查找时间复杂度和树的形态有关。
-   在最好情况下，二叉搜索树的形态与二分查找的判定树相似。每次查找都可以所辖一半搜索范围。查找路径最多从根节点到叶子节点，比较次数最多为树的高度 $\log_2 n$。在最好情况下查找的时间复杂度为 $O(\log_2 n)$。
-   在最坏情况下，二叉搜索树的形态为单支树，即只有左子树或者只有右子树。每次查找的搜索范围都缩小为 $n - 1$，退化为顺序查找，在最坏情况下时间复杂度为 $O(n)$。
-   在平均情况下，二叉搜索树的平均查找长度为 $ASL = [(n + 1) / n] * /log_2(n+1) - 1$。所以二分搜索树的查找平均时间复杂度为 $O(log_2 n)$。

### 2.2 插入

> **二叉搜索树的插入**：在二叉搜索树中插入一个值为 `val` 的节点（假设当前二叉搜索树中不存在值为 `val` 的节点）。

二叉搜索树的插入操作与二叉树的查找操作过程类似，具体步骤如下：

1.  如果二叉搜索树为空，则创建一个值为 `val` 的节点，并将其作为二叉搜索树的根节点。
2.  如果二叉搜索树不为空，则将待插入的值 `val` 与二叉搜索树根节点的值 `root.val` 进行比较：
    1.  如果 `val < root.val`，则递归将值为 `val` 的节点插入到左子树中。
    2.  如果 `val > root.val`，则递归将值为 `val` 的节点插入到右子树中。

```python
def insert_bst(self, root:TreeNode, val:int) -> TreeNode:
    """ 二叉搜索值插入
    """
    if root == None:
        return TreeNode(val)
    
    if val < root.val:
        root.left = self.insert_bst(root.left, val)
    if val > root.val:
        root.right = self.insert_bst(root.right, val)
    return root
```

### 2.3 创建

> **二叉搜索树的创建**：根据数组序列中的元素值，建立一棵二叉搜索树。

二叉搜索树的创建操作是从空树开始，按照给定数组元素的值，依次进行二叉搜索树的插入操作，最终得到一棵二叉搜索树。具体算法步骤如下：

1.  初始化二叉搜索树为空树。
2.  遍历数组元素，将数组元素值 `nums[i]` 依次插入到二叉搜索树中。
3.  将数组中全部元素值插入到二叉搜索树中之后，返回二叉搜索树的根节点。

```python
def build_bst(self, nums:list) -> TreeNode:
    """ 从列表新建二叉树
    """
    root = TreeNode()
    for n in nums:
        self.insert_bst(root, n)
    return root
```

### 2.4 删除

> **二叉搜索树的删除**：在二叉搜索树中删除值为 `val` 的节点。

在二叉搜索树中删除元素，首先要找到待删除节点，然后执行删除操作。根据待删除节点所在位置的不同，可以分为 $3$ 种情况：

1.  被删除节点的左子树为空。则令其右子树代替被删除节点的位置。
2.  被删除节点的右子树为空。则令其左子树代替被删除节点的位置。
3.  被删除节点的左右子树均不为空，则根据二叉搜索树的中序遍历有序性，删除该节点时，可以使用其直接前驱（或直接后继）代替被删除节点的位置。

-   **直接前驱**：在中序遍历中，节点 `p` 的直接前驱为其左子树的最右侧的叶子节点。
-   **直接后继**：在中序遍历中，节点 `p` 的直接后继为其右子树的最左侧的叶子节点。

二叉搜索树的删除算法步骤如下：

1.  如果当前节点为空，则返回当前节点。
2.  如果当前节点值大于 `val`，则递归去左子树中搜索并删除，此时 `root.left` 也要跟着递归更新。
3.  如果当前节点值小于 `val`，则递归去右子树中搜索并删除，此时 `root.right` 也要跟着递归更新。
4.  如果当前节点值等于 `val`，则该节点就是待删除节点。
    1.  如果当前节点的左子树为空，则删除该节点之后，则右子树代替当前节点位置，返回右子树。
    2.  如果当前节点的右子树为空，则删除该节点之后，则左子树代替当前节点位置，返回左子树。
    3.  如果当前节点的左右子树都有，则将左子树转移到右子树最左侧的叶子节点位置上，然后右子树代替当前节点位置。

```python
def delete_node(self, root:TreeNode, val:int) -> TreeNode:
    """ 删除某个节点
    """
    if not root:
        return root
    
    if root.val > val:
        root.left = self.delete_node(root.left, val)
        return root
    elif root.val < val:
        root.right = self.delete_node(root.right, val)
        return root
    else:
        # 根节点左子树为空，返回右子树
        if not root.left:
            return root.right
        # 根节点右子树为空，返回左子树
        elif not root.right:
            return root.left
        else:
            # 将root节点的左子树挂到右子树上
            ## 1.找到右子树的最小值，
            curr_node = root.right
            while curr_node.left:
                curr_node = curr_node.left
            ## 2.将根节点的左子树挂到右子树的最小值
            curr_node.left = root.left
            ## 3.返回右子树
            return root.right 

```

## 3.实战题目

### 3.1 二叉搜索树中搜索

[700. 二叉搜索树中的搜索 - 力扣（LeetCode）](https://leetcode.cn/problems/search-in-a-binary-search-tree/ "700. 二叉搜索树中的搜索 - 力扣（LeetCode）")

```python
给定二叉搜索树（BST）的根节点 root 和一个整数值 val。

你需要在 BST 中找到节点值等于 val 的节点。 返回以该节点为根的子树。 如果节点不存在，则返回 null 。

输入：root = [4,2,7,1,3], val = 2
输出：[2,1,3]

```

```c++
class Solution {
public:
    TreeNode* searchBST(TreeNode* root, int val) {
        if (root == nullptr) {
            return nullptr;
        }

        if (val == root->val) {
            return root;
        } else if (val < root->val) {
            return this->searchBST(root->left, val);
        } else {
            return this->searchBST(root->right, val);
        }
    }
};
```

### 3.2 二叉搜索树中插入节点

[701. 二叉搜索树中的插入操作 - 力扣（LeetCode）](https://leetcode.cn/problems/insert-into-a-binary-search-tree/description/ "701. 二叉搜索树中的插入操作 - 力扣（LeetCode）")

```python
给定二叉搜索树（BST）的根节点 root 和要插入树中的值 value ，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。 输入数据 保证 ，新值和原始二叉搜索树中的任意节点值都不同。

注意，可能存在多种有效的插入方式，只要树在插入后仍保持为二叉搜索树即可。 你可以返回 任意有效的结果 
```

```c++
class Solution {
public:
    TreeNode* insertIntoBST(TreeNode* root, int val) {
        if (root == nullptr) {
            return new TreeNode(val);
        }

        if (val < root->val) {
            root->left = this->insertIntoBST(root->left, val);
        } else if (val > root->val) {
            root->right = this->insertIntoBST(root->right, val);
        }

        return root;
    }
};
```

### 3.3 删除二叉搜索树中的节点

[450. 删除二叉搜索树中的节点 - 力扣（LeetCode）](https://leetcode.cn/problems/delete-node-in-a-bst/ "450. 删除二叉搜索树中的节点 - 力扣（LeetCode）")

```python
给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。

一般来说，删除节点可分为两个步骤：

1.首先找到需要删除的节点；
2.如果找到了，删除它。
 
```

1.  如果当前节点为空，则返回当前节点。
2.  如果当前节点值大于 `val`，则递归去左子树中搜索并删除，此时 `root.left` 也要跟着递归更新。
3.  如果当前节点值小于 `val`，则递归去右子树中搜索并删除，此时 `root.right` 也要跟着递归更新。
4.  如果当前节点值等于 `val`，则该节点就是待删除节点。
    1.  如果当前节点的左子树为空，则删除该节点之后，则右子树代替当前节点位置，返回右子树。
    2.  如果当前节点的右子树为空，则删除该节点之后，则左子树代替当前节点位置，返回左子树。
    3.  如果当前节点的左右子树都有，则将左子树转移到右子树最左侧的叶子节点位置上，然后右子树代替当前节点位置。

```c++
class Solution {
public:
    TreeNode* deleteNode(TreeNode* root, int key) {
        if (root == nullptr) {
            return root;
        }

        if (root->val > key) {
            root->left = this->deleteNode(root->left, key);
            return root;
        } else if (root->val < key) {
            root->right = this->deleteNode(root->right, key);
            return root;
        } else {
            // 根节点左子树为空，返回右子树
            if (root->left == nullptr) {
                return root->right;
            } else if(root->right == nullptr) {
                // 根节点右子树空，返回左子树
                return root->left;
            } else {
                // 将root节点的左子树挂在右子树的最小值上
                // 1.找到右子树的最小值
                TreeNode* curr_node = root->right;
                while (curr_node->left != nullptr) {
                    curr_node = curr_node->left;
                }
                // 2.将根节点的左子树挂到右子树的最小值
                curr_node->left = root->left;
                // 3.返回右子树
                return root->right;
            }
        }
    }
};
```

### 3.4 验证二叉搜索树

[98. 验证二叉搜索树 - 力扣（LeetCode）](https://leetcode.cn/problems/validate-binary-search-tree/description/ "98. 验证二叉搜索树 - 力扣（LeetCode）")

```python
给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。

有效 二叉搜索树定义如下：

- 节点的左子树只包含 小于 当前节点的数。
- 节点的右子树只包含 大于 当前节点的数。
- 所有左子树和右子树自身必须也是二叉搜索树。
```

中序遍历，记录父亲节点

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

### 3.5 将有序数组转化为二叉树

[108. 将有序数组转换为二叉搜索树 - 力扣（LeetCode）](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/description/ "108. 将有序数组转换为二叉搜索树 - 力扣（LeetCode）")

```python
给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。

高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。
```

二分思想，构建二叉树

```c++
class Solution {
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return this->bin_search(nums, 0, nums.size() - 1);
    }

    TreeNode* bin_search(std::vector<int>& nums, int left, int right) {
        if (left > right) {
            return nullptr;
        }
        
        int mid = left + (right - left) / 2;

        TreeNode* root = new TreeNode(nums[mid]);
        root->left = this->bin_search(nums, left, mid - 1);
        root->right = this->bin_search(nums, mid + 1, right);

        return root;
    }
};
```

### 3.6 二叉搜索树最近公共祖先

[235. 二叉搜索树的最近公共祖先 - 力扣（LeetCode）](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-search-tree/description/ "235. 二叉搜索树的最近公共祖先 - 力扣（LeetCode）")

```python
给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”


输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
输出: 6 
解释: 节点 2 和节点 8 的最近公共祖先是 6。

```

对于节点 `p`、节点 `q`，最近公共祖先就是从根节点分别到它们路径上的分岔点，也是路径中最后一个相同的节点，现在的问题就是求这个分岔点。

可以使用递归遍历查找二叉搜索树的最近公共祖先，具体方法如下。

1.  从根节点 `root` 开始遍历。
2.  如果当前节点的值大于 `p`、`q` 的值，说明 `p` 和 `q` 应该在当前节点的左子树，因此将当前节点移动到它的左子节点，继续遍历；
3.  如果当前节点的值小于 `p`、`q` 的值，说明 `p` 和 `q` 应该在当前节点的右子树，因此将当前节点移动到它的右子节点，继续遍历；
4.  如果当前节点不满足上面两种情况，则说明 `p` 和 `q` 分别在当前节点的左右子树上，则当前节点就是分岔点，直接返回该节点即可。

```c++
class Solution {
public:
    // 1.迭代实现
    TreeNode* lowestCommonAncestor1(TreeNode* root, TreeNode* p, TreeNode* q) {
        TreeNode* ancestor = root;

        while (true) {
            if (ancestor->val > p->val && ancestor->val > q->val) {
                ancestor = ancestor->left;
            } else if (ancestor->val < p->val && ancestor->val < q->val) {
                ancestor = ancestor->right;
            } else {
                break;
            }
        }

        return ancestor;
    }
    // 2.递归实现
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        // p,q都在左子树
        if (root->val > p->val && root->val > q->val) {
            return this->lowestCommonAncestor(root->left, p, q);
        } else if (root->val < p->val && root->val < q->val) {
            // pq，都在右子树
            return this->lowestCommonAncestor(root->right, p, q);
        }

        return root;
    }
};
```
