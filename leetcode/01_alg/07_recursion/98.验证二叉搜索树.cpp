/*
 * @lc app=leetcode.cn id=98 lang=cpp
 *
 * [98] 验证二叉搜索树
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

// 给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。

// 有效 二叉搜索树定义如下：

// 节点的左子树只包含 小于 当前节点的数。
// 节点的右子树只包含 大于 当前节点的数。
// 所有左子树和右子树自身必须也是二叉搜索树。

class Solution {
public:
    // BST --> 中序遍历是递增的
    bool isValidBST(TreeNode* root) {
        if (root == nullptr) {
            return true;
        }

        // 遍历左子树
        if (!this->isValidBST(root->left)) {
            return false;
        }

        // 当前结点不大于父节点，不是排序二叉树
        if (root->val <= m_last) {
            return false;
        } else {
            // 记录父节点值
            m_last = root->val;
        }
        // 遍历右子树
        if (!this->isValidBST(root->right)) {
            return false;
        }

        // 子树遍历完成，或者不是二叉排序树，退出
        return true;
    }
private:
    // 用于存储最新遍历的父亲结点值
    long m_last = LONG_MIN;
};
// @lc code=end

