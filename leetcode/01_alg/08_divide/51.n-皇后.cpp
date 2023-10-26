/*
 * @lc app=leetcode.cn id=51 lang=cpp
 *
 * [51] N 皇后
 *
 * https://leetcode.cn/problems/n-queens/description/
 *
 * algorithms
 * Hard (73.96%)
 * Likes:    1943
 * Dislikes: 0
 * Total Accepted:    342.2K
 * Total Submissions: 462.7K
 * Testcase Example:  '4'
 *
 * 按照国际象棋的规则，皇后可以攻击与之处在同一行或同一列或同一斜线上的棋子。
 * 
 * n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
 * 
 * 给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。
 * 
 * 
 * 
 * 每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：n = 4
 * 输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
 * 解释：如上图所示，4 皇后问题存在两个不同的解法。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：n = 1
 * 输出：[["Q"]]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= n <= 9
 * 
 * 
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    vector<vector<string>> solveNQueens(int n) {
        std::vector<std::string> curr_state(n, std::string(n, '.'));
        this->dfs(curr_state, 0, n);

        return m_ans;
    }

private:
    std::vector<std::vector<std::string>> m_ans;

    void dfs(std::vector<std::string>& curr_state, int row, int n) {
        // 终止条件
        if (row == n) {
            m_ans.push_back(curr_state);
            return;
        }

        // 循环每列，
        for (int col = 0; col < n; col++) {
            if (this->is_valid(curr_state, row, col)) {
                // doing
                curr_state[row][col] = 'Q';
                // 下一行
                this->dfs(curr_state, row + 1, n);
                // reverse
                curr_state[row][col] = '.';
            }
        }
    }

    bool is_valid(const std::vector<std::string>& curr_state, int row, int col) {
        int n = curr_state.size();

        // 上
        for (int i = 0; i < n; i++) {
            if (curr_state[i][col] == 'Q') {
                return false;
            }
        }

        // 左上
        for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) {
            if (curr_state[i][j] == 'Q') {
                return false;
            }
        }

        // 右上
        for (int i = row, j = col; i >= 0 && j < n; i--, j++) {
            if (curr_state[i][j] == 'Q') {
                return false;
            }
        }

        // 其他方向都是未放过皇后的，不可能为false
        return true;
    }

};
// @lc code=end

