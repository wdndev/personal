/*
 * @lc app=leetcode.cn id=576 lang=cpp
 *
 * [576] 出界的路径数
 *
 * https://leetcode.cn/problems/out-of-boundary-paths/description/
 *
 * algorithms
 * Medium (47.09%)
 * Likes:    298
 * Dislikes: 0
 * Total Accepted:    38.3K
 * Total Submissions: 81.2K
 * Testcase Example:  '2\n2\n2\n0\n0'
 *
 * 给你一个大小为 m x n 的网格和一个球。球的起始坐标为 [startRow, startColumn]
 * 。你可以将球移到在四个方向上相邻的单元格内（可以穿过网格边界到达网格之外）。你 最多 可以移动 maxMove 次球。
 * 
 * 给你五个整数 m、n、maxMove、startRow 以及 startColumn ，找出并返回可以将球移出边界的路径数量。因为答案可能非常大，返回对
 * 10^9 + 7 取余 后的结果。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：m = 2, n = 2, maxMove = 2, startRow = 0, startColumn = 0
 * 输出：6
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：m = 1, n = 3, maxMove = 3, startRow = 0, startColumn = 1
 * 输出：12
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= m, n <= 50
 * 0 <= maxMove <= 50
 * 0 <= startRow < m
 * 0 <= startColumn < n
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.记忆化搜索
    int findPaths1(int m, int n, int maxMove, int startRow, int startColumn) {
        std::vector<std::vector<std::vector<int>>> memo(m, 
            std::vector<std::vector<int>>(n, std::vector<int>(maxMove + 1, -1)));

        return this->dfs(startRow, startColumn, maxMove, memo);
    }

    // 2.动态规划
    int findPaths(int m, int n, int maxMove, int startRow, int startColumn) {
        std::vector<std::vector<std::vector<int>>> dp(m, 
            std::vector<std::vector<int>>(n, std::vector<int>(maxMove + 1, 0)));

        for (int k = 1; k <= maxMove; k++) {
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    for (int idx = 0; idx < 4; idx++) {
                        int x = i + m_dx[idx];
                        int y = j + m_dy[idx];
                        if (x >= 0 && x < m && y >= 0 && y < n) {
                            dp[i][j][k] = (dp[i][j][k] + dp[x][y][k-1]) % m_mod;
                        } else {
                            dp[i][j][k] = (dp[i][j][k] + 1) % m_mod;
                        }
                    }
                }
            }
        }

        return dp[startRow][startColumn][maxMove];
    }
private :
    // 方向
    int m_dx[4] = {-1, 1, 0, 0};
    int m_dy[4] = {0, 0, -1, 1};
    long long m_mod = 1e9 + 7;

    int dfs(int i, int j, int move_count, std::vector<std::vector<std::vector<int>>>& memo) {
        if (i < 0 || i >= memo.size() || j < 0 || j >= memo[i].size()) {
            return 1;
        }
        if (move_count == 0) {
            return 0;
        }

        if (memo[i][j][move_count] != -1) {
            return memo[i][j][move_count];
        }

        int ans = 0;
        for (int k = 0; k < 4; k++) {
            int x = i + m_dx[k];
            int y = j + m_dy[k];

            ans += this->dfs(x, y, move_count - 1, memo);
            ans %= m_mod;
        }
        memo[i][j][move_count] = ans;

        return ans;
    }

};
// @lc code=end

