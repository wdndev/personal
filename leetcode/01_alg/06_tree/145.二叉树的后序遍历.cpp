/*
 * @lc app=leetcode.cn id=145 lang=cpp
 *
 * [145] 二叉树的后序遍历
 *
 * https://leetcode.cn/problems/binary-tree-postorder-traversal/description/
 *
 * algorithms
 * Easy (76.30%)
 * Likes:    1109
 * Dislikes: 0
 * Total Accepted:    686K
 * Total Submissions: 898.9K
 * Testcase Example:  '[1,null,2,3]'
 *
 * 给你一棵二叉树的根节点 root ，返回其节点值的 后序遍历 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：root = [1,null,2,3]
 * 输出：[3,2,1]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：root = []
 * 输出：[]
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：root = [1]
 * 输出：[1]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 树中节点的数目在范围 [0, 100] 内
 * -100 <= Node.val <= 100
 * 
 * 
 * 
 * 
 * 进阶：递归算法很简单，你可以通过迭代算法完成吗？
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
    vector<int> postorderTraversal1(TreeNode* root) {
        std::vector<int> ans;
        this->postorder(root, ans);

        return ans;
    }

    vector<int> postorderTraversal(TreeNode* root) {
        std::vector<int> ans;
        
        if (root == nullptr) {
            return ans;
        }

        std::stack<TreeNode*> stack;
        TreeNode* prev = nullptr;

        while (!stack.empty() || root != nullptr) {
            while (root != nullptr) {
                stack.push(root);
                root = root->left;
            }

            root = stack.top();
            stack.pop();

            if (root->right == nullptr || root->right == prev) {
                ans.emplace_back(root->val);
                prev = root;
                root = nullptr;
            } else {
                stack.push(root);
                root = root->right;
            }
        }

        return ans;
    }

private:
    void postorder(TreeNode* root, std::vector<int>& ans) {
        if (!root) {
            return;
        }
        this->postorder(root->left, ans);
        this->postorder(root->right, ans);
        ans.push_back(root->val);
    }
};
// @lc code=end

