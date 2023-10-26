/*
 * @lc app=leetcode.cn id=77 lang=cpp
 *
 * [77] 组合
 *
 * https://leetcode.cn/problems/combinations/description/
 *
 * algorithms
 * Medium (77.07%)
 * Likes:    1515
 * Dislikes: 0
 * Total Accepted:    608.3K
 * Total Submissions: 789.3K
 * Testcase Example:  '4\n2'
 *
 * 给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。
 * 
 * 你可以按 任何顺序 返回答案。
 * 
 * 示例 1：
 * 输入：n = 4, k = 2
 * 输出：
 * [
 * ⁠ [2,4],
 * ⁠ [3,4],
 * ⁠ [2,3],
 * ⁠ [1,2],
 * ⁠ [1,3],
 * ⁠ [1,4],
 * ]
 * 
 * 示例 2：
 * 输入：n = 1, k = 1
 * 输出：[[1]]
 * 
 * 
 * 提示：
 * 
 * 
 * 1 
 * 1 
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    vector<vector<int>> combine(int n, int k) {
        std::vector<std::vector<int>> ans;
        this->dfs(1, n, k, ans);
        return ans;
    }

private:
    void dfs(int curr, int n, int k, std::vector<std::vector<int>>& ans) {
        // 终止条件
        // tmp长度加上区间[cur, n]的长度小于k，不可能构造出长度k的tmp
        if (m_tmp.size() + (n - curr + 1) < k) {
            return;
        }
        // 记录合法的答案
        if (m_tmp.size() == k) {
            ans.push_back(m_tmp);
            return;
        }

        // 处理当前层
        m_tmp.push_back(curr);
        this->dfs(curr + 1, n, k, ans);
        m_tmp.pop_back();
        this->dfs(curr + 1, n, k, ans);
    }

    std::vector<int> m_tmp;
};
// @lc code=end

