/*
 * @lc app=leetcode.cn id=700 lang=cpp
 *
 * [700] 二叉搜索树中的搜索
 *
 * https://leetcode.cn/problems/search-in-a-binary-search-tree/description/
 *
 * algorithms
 * Easy (77.87%)
 * Likes:    458
 * Dislikes: 0
 * Total Accepted:    298.1K
 * Total Submissions: 382.7K
 * Testcase Example:  '[4,2,7,1,3]\n2'
 *
 * 给定二叉搜索树（BST）的根节点 root 和一个整数值 val。
 * 
 * 你需要在 BST 中找到节点值等于 val 的节点。 返回以该节点为根的子树。 如果节点不存在，则返回 null 。
 * 
 * 
 * 
 * 示例 1:
 * 
 * 
 * 
 * 
 * 输入：root = [4,2,7,1,3], val = 2
 * 输出：[2,1,3]
 * 
 * 
 * 示例 2:
 * 
 * 
 * 输入：root = [4,2,7,1,3], val = 5
 * 输出：[]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 树中节点数在 [1, 5000] 范围内
 * 1 <= Node.val <= 10^7
 * root 是二叉搜索树
 * 1 <= val <= 10^7
 * 
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
// @lc code=end

