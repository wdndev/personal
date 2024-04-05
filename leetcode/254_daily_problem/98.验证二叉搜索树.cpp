/*
 * @lc app=leetcode.cn id=98 lang=cpp
 *
 * [98] 验证二叉搜索树
 *
 * https://leetcode.cn/problems/validate-binary-search-tree/description/
 *
 * algorithms
 * Medium (37.69%)
 * Likes:    2313
 * Dislikes: 0
 * Total Accepted:    883.8K
 * Total Submissions: 2.3M
 * Testcase Example:  '[2,1,3]'
 *
 * 给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。
 * 
 * 有效 二叉搜索树定义如下：
 * 
 * 
 * 节点的左子树只包含 小于 当前节点的数。
 * 节点的右子树只包含 大于 当前节点的数。
 * 所有左子树和右子树自身必须也是二叉搜索树。
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：root = [2,1,3]
 * 输出：true
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：root = [5,1,4,null,null,3,6]
 * 输出：false
 * 解释：根节点的值是 5 ，但是右子节点的值是 4 。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 树中节点数目范围在[1, 10^4] 内
 * -2^31 <= Node.val <= 2^31 - 1
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
    // // 1.前序遍历
    // bool isValidBST(TreeNode* root, long left = LONG_MIN, long right = LONG_MAX) {
    //     if (root == nullptr) {
    //         return true;
    //     }
    //     long x = root->val;
    //     return left < x && x < right && 
    //            this->isValidBST(root->left, left, x) &&
    //            this->isValidBST(root->right, x, right);
    // }

    // // 2.中序遍历
    // long m_pre = LONG_MIN;
    // bool isValidBST(TreeNode* root) {
    //     if (root == nullptr) {
    //         return true;
    //     }
    //     if (!this->isValidBST(root->left) || root->val <= m_pre) {
    //         return false;
    //     }
    //     m_pre = root->val;
    //     return this->isValidBST(root->right);
    // }

    // 3.后序遍历
    std::pair<long, long> dfs(TreeNode* node) {
        if (node == nullptr) {
            return {LONG_MAX, LONG_MIN};
        }
        auto[l_min, l_max] = this->dfs(node->left);
        auto[r_min, r_max] = this->dfs(node->right);
        long x = node->val;
        // 判断是不是二叉搜索树
        if (x <= l_max || x >= r_min) {
            return {LONG_MIN, LONG_MAX};
        }
        return {std::min(l_min, x), std::max(r_max, x)};
    }

    bool isValidBST(TreeNode* root) {
        return this->dfs(root).second != LONG_MAX;
    }
};
// @lc code=end

