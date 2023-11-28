/*
 * @lc app=leetcode.cn id=199 lang=cpp
 *
 * [199] 二叉树的右视图
 *
 * https://leetcode.cn/problems/binary-tree-right-side-view/description/
 *
 * algorithms
 * Medium (66.30%)
 * Likes:    995
 * Dislikes: 0
 * Total Accepted:    342.2K
 * Total Submissions: 516.1K
 * Testcase Example:  '[1,2,3,null,5,null,4]'
 *
 * 给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
 * 
 * 
 * 
 * 示例 1:
 * 
 * 
 * 
 * 
 * 输入: [1,2,3,null,5,null,4]
 * 输出: [1,3,4]
 * 
 * 
 * 示例 2:
 * 
 * 
 * 输入: [1,null,3]
 * 输出: [1,3]
 * 
 * 
 * 示例 3:
 * 
 * 
 * 输入: []
 * 输出: []
 * 
 * 
 * 
 * 
 * 提示:
 * 
 * 
 * 二叉树的节点个数的范围是 [0,100]
 * -100  
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
    // 层序遍历，每一层的最后一个
    vector<int> rightSideView(TreeNode* root) {
        std::vector<int> ans;
        if (root == nullptr) {
            return ans;
        }

        std::queue<TreeNode*> que;
        que.push(root);

        while(!que.empty()) {
            std::vector<int> level_node;
            int level_size = que.size();
            for (int i = 0; i < level_size; i++) {
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
            ans.push_back(level_node[level_size - 1]);
        }

        return ans;
    }
};
// @lc code=end

