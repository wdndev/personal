/*
 * @lc app=leetcode.cn id=235 lang=cpp
 *
 * [235] 二叉搜索树的最近公共祖先
 *
 * https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-search-tree/description/
 *
 * algorithms
 * Medium (69.15%)
 * Likes:    1191
 * Dislikes: 0
 * Total Accepted:    384.7K
 * Total Submissions: 556K
 * Testcase Example:  '[6,2,8,0,4,7,9,null,null,3,5]\n2\n8'
 *
 * 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
 * 
 * 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x
 * 的深度尽可能大（一个节点也可以是它自己的祖先）。”
 * 
 * 例如，给定如下二叉搜索树:  root = [6,2,8,0,4,7,9,null,null,3,5]
 * 
 * 
 * 
 * 
 * 
 * 示例 1:
 * 
 * 输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
 * 输出: 6 
 * 解释: 节点 2 和节点 8 的最近公共祖先是 6。
 * 
 * 
 * 示例 2:
 * 
 * 输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
 * 输出: 2
 * 解释: 节点 2 和节点 4 的最近公共祖先是 2, 因为根据定义最近公共祖先节点可以为节点本身。
 * 
 * 
 * 
 * 说明:
 * 
 * 
 * 所有节点的值都是唯一的。
 * p、q 为不同节点且均存在于给定的二叉搜索树中。
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
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */

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
// @lc code=end

