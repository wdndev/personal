/*
 * @lc app=leetcode.cn id=64 lang=cpp
 *
 * [64] 最小路径和
 *
 * https://leetcode.cn/problems/minimum-path-sum/description/
 *
 * algorithms
 * Medium (69.81%)
 * Likes:    1611
 * Dislikes: 0
 * Total Accepted:    539.8K
 * Total Submissions: 773.2K
 * Testcase Example:  '[[1,3,1],[1,5,1],[4,2,1]]'
 *
 * 给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
 * 
 * 说明：每次只能向下或者向右移动一步。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
 * 输出：7
 * 解释：因为路径 1→3→1→1→1 的总和最小。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：grid = [[1,2,3],[4,5,6]]
 * 输出：12
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * m == grid.length
 * n == grid[i].length
 * 1 <= m, n <= 200
 * 0 <= grid[i][j] <= 200
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.动态规划
    // 状态定义：dp[i][j] : 从00出发到i，j的最短路径
    // 状态转移方程：dp[i][j] = std::min(dp[i - 1][j], dp[i][j - 1]) + grid[i - 1][j - 1];
    int minPathSum(vector<vector<int>>& grid) {
        int row = grid.size();
        int col = grid[0].size();

        std::vector<std::vector<int>> dp(row + 1, std::vector<int>(col + 1, INT_MAX));
        dp[0][1] = 0;

        for (int i = 1; i <= row; i++) {
            for (int j = 1; j <= col; j++) {
                // 注意dp和grid的索引不一样
                dp[i][j] = std::min(dp[i - 1][j], dp[i][j - 1]) + grid[i - 1][j - 1];
            }
        }
        return dp[row][col];
    }
};
// @lc code=end

