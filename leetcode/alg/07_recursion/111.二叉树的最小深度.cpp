/*
 * @lc app=leetcode.cn id=111 lang=cpp
 *
 * [111] 二叉树的最小深度
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

// 给定一个二叉树，找出其最小深度。
// 最小深度是从根节点到最近叶子节点的最短路径上的节点数量。
// 说明：叶子节点是指没有子节点的节点。


class Solution {
public:
    int minDepth(TreeNode* root) {
        // 当 root 为空时，返回 0
        if (!root) {
            return 0;
        }

        int deep = 0;
        int left_deep = minDepth(root->left);
        int right_deep = minDepth(root->right);
        // 当 root 节点左右孩子有一个为空时，返回不为空的孩子节点的深度
        // 当 root 节点左右孩子都不为空时，返回左右孩子较小深度的节点值
        if (!root->left || !root->right) {
            deep = left_deep + right_deep + 1;
        } else {
            deep = std::min(left_deep, right_deep) + 1;
        }

        return deep;
    }
};
// @lc code=end

