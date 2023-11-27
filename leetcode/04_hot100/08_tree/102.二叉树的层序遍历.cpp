/*
 * @lc app=leetcode.cn id=102 lang=cpp
 *
 * [102] 二叉树的层序遍历
 *
 * https://leetcode.cn/problems/binary-tree-level-order-traversal/description/
 *
 * algorithms
 * Medium (66.05%)
 * Likes:    1836
 * Dislikes: 0
 * Total Accepted:    900.5K
 * Total Submissions: 1.4M
 * Testcase Example:  '[3,9,20,null,null,15,7]'
 *
 * 给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：root = [3,9,20,null,null,15,7]
 * 输出：[[3],[9,20],[15,7]]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：root = [1]
 * 输出：[[1]]
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：root = []
 * 输出：[]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 树中节点数目在范围 [0, 2000] 内
 * -1000 <= Node.val <= 1000
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
    // 1.广度优先遍历
    vector<vector<int>> levelOrder1(TreeNode* root) {
        std::vector<std::vector<int>> ans;
        if (root == nullptr) {
            return ans;
        }

        std::queue<TreeNode*> que;
        que.push(root);

        while (!que.empty()) {
            std::vector<int> level_node;
            int node_count = que.size();
            for (int i = 0; i < node_count; i++) 
            {
                TreeNode* tmp_node = que.front();
                que.pop();

                level_node.push_back(tmp_node->val);

                if (tmp_node->left != nullptr) {
                    que.push(tmp_node->left);
                }

                if (tmp_node->right != nullptr) {
                    que.push(tmp_node->right);
                }
            }
            
            ans.push_back(std::move(level_node));
        }

        return ans;
    }

    // 2.深度优先遍历
    vector<vector<int>> levelOrder(TreeNode* root) {
        std::vector<std::vector<int>> ans;
        if (root == nullptr) {
            return ans;
        }

        this->dfs(root, 0, ans);

        return ans;
    }

    void dfs(TreeNode* root, int depth, std::vector<std::vector<int>>& ans) {
        if (root == nullptr) {
            return;
        }

        if (depth >= ans.size()) {
            ans.push_back(std::vector<int>{});
        }
        ans[depth].push_back(root->val);

        this->dfs(root->left, depth + 1, ans);
        this->dfs(root->right, depth + 1, ans);
    }
};
// @lc code=end

