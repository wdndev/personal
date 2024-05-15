/*
 * @lc app=leetcode.cn id=530 lang=cpp
 *
 * [530] 二叉搜索树的最小绝对差
 *
 * https://leetcode.cn/problems/minimum-absolute-difference-in-bst/description/
 *
 * algorithms
 * Easy (63.25%)
 * Likes:    512
 * Dislikes: 0
 * Total Accepted:    201.7K
 * Total Submissions: 318.9K
 * Testcase Example:  '[4,2,6,1,3]'
 *
 * 给你一个二叉搜索树的根节点 root ，返回 树中任意两不同节点值之间的最小差值 。
 * 
 * 差值是一个正数，其数值等于两值之差的绝对值。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：root = [4,2,6,1,3]
 * 输出：1
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：root = [1,0,48,null,null,12,49]
 * 输出：1
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 树中节点的数目范围是 [2, 10^4]
 * 0 <= Node.val <= 10^5
 * 
 * 
 * 
 * 
 * 注意：本题与 783
 * https://leetcode-cn.com/problems/minimum-distance-between-bst-nodes/ 相同
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
    // 二叉树中序遍历
    int getMinimumDifference(TreeNode* root) {
        if (root == nullptr) {
            return 0;
        }
        std::stack<TreeNode*> stk;

        int pre_val = -1;
        int min_val = INT_MAX;
        while (root != nullptr || !stk.empty()) {
            // 左子树入栈
            while (root != nullptr) {
                stk.push(root);
                root = root->left;
            }
            // 从栈中取出
            root = stk.top();
            stk.pop();

            std::cout << root->val << std::endl;
            if (pre_val == -1) {
                pre_val = root->val;
            } else {
                int diff = abs(root->val - pre_val);
                min_val = min(min_val, diff);
                pre_val = root->val;
            }
            

            // 遍历右子树
            root = root->right;
        }

        return min_val;
    }
};
// @lc code=end

