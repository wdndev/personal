/*
 * @lc app=leetcode.cn id=114 lang=cpp
 *
 * [114] 二叉树展开为链表
 *
 * https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/description/
 *
 * algorithms
 * Medium (73.22%)
 * Likes:    1587
 * Dislikes: 0
 * Total Accepted:    404.3K
 * Total Submissions: 552.1K
 * Testcase Example:  '[1,2,5,3,4,null,6]'
 *
 * 给你二叉树的根结点 root ，请你将它展开为一个单链表：
 * 
 * 
 * 展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
 * 展开后的单链表应该与二叉树 先序遍历 顺序相同。
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：root = [1,2,5,3,4,null,6]
 * 输出：[1,null,2,null,3,null,4,null,5,null,6]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：root = []
 * 输出：[]
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：root = [0]
 * 输出：[0]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 树中结点数在范围 [0, 2000] 内
 * -100 
 * 
 * 
 * 
 * 
 * 进阶：你可以使用原地算法（O(1) 额外空间）展开这棵树吗？
 * 
 */

// @lc code=start
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
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
// @lc code=end

