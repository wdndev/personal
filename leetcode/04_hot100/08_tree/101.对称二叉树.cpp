/*
 * @lc app=leetcode.cn id=101 lang=cpp
 *
 * [101] 对称二叉树
 *
 * https://leetcode.cn/problems/symmetric-tree/description/
 *
 * algorithms
 * Easy (59.35%)
 * Likes:    2597
 * Dislikes: 0
 * Total Accepted:    934.3K
 * Total Submissions: 1.6M
 * Testcase Example:  '[1,2,2,3,4,4,3]'
 *
 * 给你一个二叉树的根节点 root ， 检查它是否轴对称。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：root = [1,2,2,3,4,4,3]
 * 输出：true
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：root = [1,2,2,null,3,null,3]
 * 输出：false
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 树中节点数目在范围 [1, 1000] 内
 * -100 <= Node.val <= 100
 * 
 * 
 * 
 * 
 * 进阶：你可以运用递归和迭代两种方法解决这个问题吗？
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
// @lc code=end

