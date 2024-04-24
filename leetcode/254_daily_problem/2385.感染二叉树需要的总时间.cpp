/*
 * @lc app=leetcode.cn id=2385 lang=cpp
 *
 * [2385] 感染二叉树需要的总时间
 *
 * https://leetcode.cn/problems/amount-of-time-for-binary-tree-to-be-infected/description/
 *
 * algorithms
 * Medium (47.05%)
 * Likes:    94
 * Dislikes: 0
 * Total Accepted:    21.1K
 * Total Submissions: 41.6K
 * Testcase Example:  '[1,5,3,null,4,10,6,9,2]\n3'
 *
 * 给你一棵二叉树的根节点 root ，二叉树中节点的值 互不相同 。另给你一个整数 start 。在第 0 分钟，感染 将会从值为 start
 * 的节点开始爆发。
 * 
 * 每分钟，如果节点满足以下全部条件，就会被感染：
 * 
 * 
 * 节点此前还没有感染。
 * 节点与一个已感染节点相邻。
 * 
 * 
 * 返回感染整棵树需要的分钟数。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 输入：root = [1,5,3,null,4,10,6,9,2], start = 3
 * 输出：4
 * 解释：节点按以下过程被感染：
 * - 第 0 分钟：节点 3
 * - 第 1 分钟：节点 1、10、6
 * - 第 2 分钟：节点5
 * - 第 3 分钟：节点 4
 * - 第 4 分钟：节点 9 和 2
 * 感染整棵树需要 4 分钟，所以返回 4 。
 * 
 * 
 * 示例 2：
 * 
 * 输入：root = [1], start = 1
 * 输出：0
 * 解释：第 0 分钟，树中唯一一个节点处于感染状态，返回 0 。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 树中节点的数目在范围 [1, 10^5] 内
 * 1 <= Node.val <= 10^5
 * 每个节点的值 互不相同
 * 树中必定存在值为 start 的节点
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
    // 记录父节点 + DFS
    // start_node 出发 dfs，求出二叉树的最大深度，即为答案。
    // 注意：除了递归左右儿子外，还需要递归父节点。
    // 为了避免重复访问父节点，添加一个参数 from，表示当前节点从节点from过来
    int amountOfTime(TreeNode* root, int start) {
        this->m_start = start;
        this->dfs(root, nullptr);
        return this->max_depth(m_start_node, m_start_node);
    }
private:
    // 开始节点值
    int m_start;
    // 开始节点
    TreeNode* m_start_node;
    // 哈希表
    TreeNode* m_hash[100001];

    void dfs(TreeNode* node, TreeNode* from) {
        if (node == nullptr) {
            return;
        }
        // 记录每个节点的父节点
        m_hash[node->val] = from;
        // 找到 start
        if (node->val == m_start) {
            m_start_node = node;
        }
        this->dfs(node->left, node);
        this->dfs(node->right, node);
    }

    int max_depth(TreeNode* node, TreeNode* from) {
        // 注意这里是 -1，因为 start 的深度为 0
        if (node == nullptr) {
            return -1;
        }
        int res = -1;
        if (node->left != from) {
            res = std::max(res, this->max_depth(node->left, node));
        }
        if (node->right != from) {
            res = std::max(res, this->max_depth(node->right, node));
        }
        if (m_hash[node->val] != from) {
            res = std::max(res, this->max_depth(m_hash[node->val], node));
        }

        return res + 1;
    }   
};
// @lc code=end

