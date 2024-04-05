/*
 * @lc app=leetcode.cn id=1026 lang=cpp
 *
 * [1026] 节点与其祖先之间的最大差值
 *
 * https://leetcode.cn/problems/maximum-difference-between-node-and-ancestor/description/
 *
 * algorithms
 * Medium (75.52%)
 * Likes:    233
 * Dislikes: 0
 * Total Accepted:    41.9K
 * Total Submissions: 54.8K
 * Testcase Example:  '[8,3,10,1,6,null,14,null,null,4,7,13]'
 *
 * 给定二叉树的根节点 root，找出存在于 不同 节点 A 和 B 之间的最大值 V，其中 V = |A.val - B.val|，且 A 是 B
 * 的祖先。
 * 
 * （如果 A 的任何子节点之一为 B，或者 A 的任何子节点是 B 的祖先，那么我们认为 A 是 B 的祖先）
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 
 * 
 * 输入：root = [8,3,10,1,6,null,14,null,null,4,7,13]
 * 输出：7
 * 解释： 
 * 我们有大量的节点与其祖先的差值，其中一些如下：
 * |8 - 3| = 5
 * |3 - 7| = 4
 * |8 - 1| = 7
 * |10 - 13| = 3
 * 在所有可能的差值中，最大值 7 由 |8 - 1| = 7 得出。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：root = [1,null,2,null,0,3]
 * 输出：3
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 树中的节点数在 2 到 5000 之间。
 * 0 
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
    // 计算节点的最大差值，相同节点差值为0，所以相同节点不影响最大差值
    // 对于最大差值，无需记录路径中的全部节点，只需记录路径中的最大节点值 mx 和最小节点值 mn
    // 每递归到一个节点 B，计算 max(|mn - B.val|, |mx - B.val|)
    // 可以简化为 max(B.val - mn, mx - B.val)
    int maxAncestorDiff(TreeNode* root) {
        this->dfs_opti(root, root->val, root->val);

        return m_ans;
    }

private:
    int m_ans;

    void dfs(TreeNode* node, int mn, int mx) {
        if (node == nullptr) {
            return;
        }
        // 相同节点差值为0，不影响最大差值
        // 所以，先更新 mn 和 mx， 再计算差值也是可以的
        // 再这种情况下，一定满足 mn <= node.val <= mx
        mn = std::min(mn, node->val);
        mx = std::max(mx, node->val);
        m_ans = std::max(m_ans, std::max(node->val - mn, mx - node->val));
        this->dfs(node->left, mn, mx);
        this->dfs(node->right, mn, mx);
    }

    // 优化
    // 对于一条从根出发的向下的路径，要计算是这条路径上任意两点的最大差值
    // 递归到空结点时，mx是从根节点到叶子节点的路径上的最大值，mn 是从跟到叶子的路径上的最小值，
    // 所以 mx - mn就是从根节点到叶子路径上任意两点的最大差值
    // 所以，无需每个节点都去更新答案，而是再递归到空节点时才去更新答案
    void dfs_opti(TreeNode* node, int mn, int mx) {
        if (node == nullptr) {
            m_ans = std::max(m_ans, mx - mn);
            return;
        }
        // 相同节点差值为0，不影响最大差值
        // 所以，先更新 mn 和 mx， 再计算差值也是可以的
        // 再这种情况下，一定满足 mn <= node.val <= mx
        mn = std::min(mn, node->val);
        mx = std::max(mx, node->val);
        this->dfs(node->left, mn, mx);
        this->dfs(node->right, mn, mx);
    }
};
// @lc code=end

