/*
 * @lc app=leetcode.cn id=230 lang=cpp
 *
 * [230] 二叉搜索树中第K小的元素
 *
 * https://leetcode.cn/problems/kth-smallest-element-in-a-bst/description/
 *
 * algorithms
 * Medium (76.45%)
 * Likes:    795
 * Dislikes: 0
 * Total Accepted:    304.5K
 * Total Submissions: 398.3K
 * Testcase Example:  '[3,1,4,null,2]\n1'
 *
 * 给定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 个最小元素（从 1 开始计数）。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：root = [3,1,4,null,2], k = 1
 * 输出：1
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：root = [5,3,6,2,4,null,null,1], k = 3
 * 输出：3
 * 
 * 
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 树中的节点数为 n 。
 * 1 
 * 0 
 * 
 * 
 * 
 * 
 * 进阶：如果二叉搜索树经常被修改（插入/删除操作）并且你需要频繁地查找第 k 小的值，你将如何优化算法？
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
    // 中序遍历
    int kthSmallest(TreeNode* root, int k) {
        std::vector<int> ans;
        
        int val = -1;
        this->inorder(root, k, val);

        return val;
    }

    void inorder(TreeNode* root, int idx, int& val) {
        if (root == nullptr) {
            return;
        }
        
        this->inorder(root->left, idx, val);
        
        m_idx++;
        if (m_idx == idx) {
            val = root->val;
            return;
        }
        // idx--;
        
        this->inorder(root->right, idx, val);
    }

    // void inorder(TreeNode* root, std::vector<int>& ans) {
    //     if (root == nullptr) {
    //         return;
    //     }
    //     this->inorder(root->left, ans);
    //     ans.push_back(root->val);
    //     this->inorder(root->right, ans);
    // }
    int m_idx = 0;
};
// @lc code=end

