/*
 * @lc app=leetcode.cn id=894 lang=cpp
 *
 * [894] 所有可能的真二叉树
 *
 * https://leetcode.cn/problems/all-possible-full-binary-trees/description/
 *
 * algorithms
 * Medium (77.66%)
 * Likes:    391
 * Dislikes: 0
 * Total Accepted:    34.3K
 * Total Submissions: 42.7K
 * Testcase Example:  '7'
 *
 * 给你一个整数 n ，请你找出所有可能含 n 个节点的 真二叉树 ，并以列表形式返回。答案中每棵树的每个节点都必须符合 Node.val == 0 。
 * 
 * 答案的每个元素都是一棵真二叉树的根节点。你可以按 任意顺序 返回最终的真二叉树列表。
 * 
 * 真二叉树 是一类二叉树，树中每个节点恰好有 0 或 2 个子节点。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：n = 7
 * 
 * 输出：[[0,0,0,null,null,0,0,null,null,0,0],[0,0,0,null,null,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,null,null,null,null,0,0],[0,0,0,0,0,null,null,0,0]]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：n = 3
 * 输出：[[0,0,0]]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= n <= 20
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
    // 由于真二叉树中的每个结点恰好有0或2个子节点，如果往一颗真二叉树上添加节点，最少添加两个，这意味着没课字数节点个数为奇数个
    // 每增加2个节点，真二叉树就会多1个叶子节点，所以一颗n个节点的真二叉树恰好有 (n+1)/2个叶子
    // 动态规划：
    // 1.状态定义：f[i]为有i个叶子节点的所有真二叉树的列表
    // 2.初始值：f[1]为只包含一个节点的二叉树列表
    // 3.递推公式：枚举左子树有j=1,2,3,...,i-1个叶子，那么右子树有i-j个叶子
    //      左子树的所有真二叉树列表为f[j], 右子树的所有真二叉树列表为f[i-j]。
    //      从这两个列表中各选一颗真二叉树，作为根节点的左右子树，从而得到有i个叶子的真二叉树，这些真二叉树组成了f[i]
    // 4.答案：
    //   - 如果 n 是偶数，返回空列表
    //   - 如果 n 是奇数，返回 f[(n+1)/2]
    vector<TreeNode*> allPossibleFBT(int n) {
        std::vector<TreeNode*> f[11];
        f[1] = {new TreeNode()};
        // 计算f
        for (int i = 2; i < 11; i++) {
            // 枚举左子树叶子数
            for (int j = 1; j < i; j++) {
                // 枚举左子树
                for (auto left : f[j]) {
                    // 枚举右子树
                    for (auto right : f[i - j]) {
                        f[i].push_back(new TreeNode(0, left, right));
                    }
                }
            }
        }

        return f[n % 2 ? (n + 1) / 2 : 0];
    }
    
};
// @lc code=end

