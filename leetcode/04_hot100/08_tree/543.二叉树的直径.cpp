/*
 * @lc app=leetcode.cn id=543 lang=cpp
 *
 * [543] 二叉树的直径
 *
 * https://leetcode.cn/problems/diameter-of-binary-tree/description/
 *
 * algorithms
 * Easy (59.19%)
 * Likes:    1443
 * Dislikes: 0
 * Total Accepted:    362.4K
 * Total Submissions: 612K
 * Testcase Example:  '[1,2,3,4,5]'
 *
 * 给你一棵二叉树的根节点，返回该树的 直径 。
 * 
 * 二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点 root 。
 * 
 * 两节点之间路径的 长度 由它们之间边数表示。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：root = [1,2,3,4,5]
 * 输出：3
 * 解释：3 ，取路径 [4,2,1,3] 或 [5,2,1,3] 的长度。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：root = [1,2]
 * 输出：1
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 树中节点数目在范围 [1, 10^4] 内
 * -100 <= Node.val <= 100
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
// @lc code=end

