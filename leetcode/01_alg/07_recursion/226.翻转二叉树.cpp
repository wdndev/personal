/*
 * @lc app=leetcode.cn id=226 lang=cpp
 *
 * [226] 翻转二叉树
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

// 给你一棵二叉树的根节点 root ，翻转这棵二叉树，并返回其根节点。

class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        // 退出递归
        if (!root) {
            return nullptr;
        }

        // 交换根节点的左右子树
        TreeNode* left = this->invertTree(root->left);
        TreeNode* right = this->invertTree(root->right);

        root->left = right;
        root->right = left;

        // 返回根节点
        return root;
    }
};
// @lc code=end

