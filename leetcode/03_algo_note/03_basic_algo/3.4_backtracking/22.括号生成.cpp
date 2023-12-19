/*
 * @lc app=leetcode.cn id=22 lang=cpp
 *
 * [22] 括号生成
 *
 * https://leetcode.cn/problems/generate-parentheses/description/
 *
 * algorithms
 * Medium (77.51%)
 * Likes:    3464
 * Dislikes: 0
 * Total Accepted:    773.1K
 * Total Submissions: 997.1K
 * Testcase Example:  '3'
 *
 * 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：n = 3
 * 输出：["((()))","(()())","(())()","()(())","()()()"]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：n = 1
 * 输出：["()"]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= n <= 8
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 递归
    // 左括号: 随时加，只要不超标
    // 右括号 : 必须之前有左括号，且左括号个数 > 右括号个数
    vector<string> generateParenthesis(int n) {
        std::vector<std::string> ans;
        std::string tmp_str;

        this->dfs(0, 0, n, tmp_str, ans);

        return ans;
    }

    void dfs(int left, int right, int n, std::string tmp_str, std::vector<std::string>& ans) {
        // 终止条件
        if (left == n && right == n) {
            ans.push_back(tmp_str);
            // std::cout << "left: " << left << " " << tmp_str << std::endl;
            return;
        }

        // 递归处理
        if (left < n) {
            this->dfs(left + 1, right, n, tmp_str + "(", ans);
        }

        if (left > right) {
            this->dfs(left, right + 1, n, tmp_str + ")", ans);
        }
    }
};
// @lc code=end

