/*
 * @lc app=leetcode.cn id=450 lang=cpp
 *
 * [450] 删除二叉搜索树中的节点
 *
 * https://leetcode.cn/problems/delete-node-in-a-bst/description/
 *
 * algorithms
 * Medium (52.29%)
 * Likes:    1296
 * Dislikes: 0
 * Total Accepted:    241.6K
 * Total Submissions: 461.8K
 * Testcase Example:  '[5,3,6,2,4,null,7]\n3'
 *
 * 给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key
 * 对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。
 * 
 * 一般来说，删除节点可分为两个步骤：
 * 
 * 
 * 首先找到需要删除的节点；
 * 如果找到了，删除它。
 * 
 * 
 * 
 * 
 * 示例 1:
 * 
 * 
 * 
 * 
 * 输入：root = [5,3,6,2,4,null,7], key = 3
 * 输出：[5,4,6,2,null,null,7]
 * 解释：给定需要删除的节点值是 3，所以我们首先找到 3 这个节点，然后删除它。
 * 一个正确的答案是 [5,4,6,2,null,null,7], 如下图所示。
 * 另一个正确答案是 [5,2,6,null,4,null,7]。
 * 
 * 
 * 
 * 
 * 示例 2:
 * 
 * 
 * 输入: root = [5,3,6,2,4,null,7], key = 0
 * 输出: [5,3,6,2,4,null,7]
 * 解释: 二叉树不包含值为 0 的节点
 * 
 * 
 * 示例 3:
 * 
 * 
 * 输入: root = [], key = 0
 * 输出: []
 * 
 * 
 * 
 * 提示:
 * 
 * 
 * 节点数的范围 [0, 10^4].
 * -10^5 <= Node.val <= 10^5
 * 节点值唯一
 * root 是合法的二叉搜索树
 * -10^5 <= key <= 10^5
 * 
 * 
 * 
 * 
 * 进阶： 要求算法时间复杂度为 O(h)，h 为树的高度。
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
// @lc code=end

