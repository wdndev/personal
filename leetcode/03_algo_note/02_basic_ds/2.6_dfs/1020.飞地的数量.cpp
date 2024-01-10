/*
 * @lc app=leetcode.cn id=1020 lang=cpp
 *
 * [1020] 飞地的数量
 *
 * https://leetcode.cn/problems/number-of-enclaves/description/
 *
 * algorithms
 * Medium (61.93%)
 * Likes:    255
 * Dislikes: 0
 * Total Accepted:    70.5K
 * Total Submissions: 113.9K
 * Testcase Example:  '[[0,0,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,0]]'
 *
 * 给你一个大小为 m x n 的二进制矩阵 grid ，其中 0 表示一个海洋单元格、1 表示一个陆地单元格。
 * 
 * 一次 移动 是指从一个陆地单元格走到另一个相邻（上、下、左、右）的陆地单元格或跨过 grid 的边界。
 * 
 * 返回网格中 无法 在任意次数的移动中离开网格边界的陆地单元格的数量。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：grid = [[0,0,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,0]]
 * 输出：3
 * 解释：有三个 1 被 0 包围。一个 1 没有被包围，因为它在边界上。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：grid = [[0,1,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,0]]
 * 输出：0
 * 解释：所有 1 都在边界上或可以到达边界。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * m == grid.length
 * n == grid[i].length
 * 1 <= m, n <= 500
 * grid[i][j] 的值为 0 或 1
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    int numEnclaves(vector<vector<int>>& grid) {
        int rows = grid.size();
        int cols = grid[0].size();

        for (int i = 0; i < rows; i++) {
            if (grid[i][0] == 1) {
                this->dfs(i, 0, grid);
            }
            if (grid[i][cols - 1] == 1) {
                this->dfs(i, cols - 1, grid);
            }
        }

        for (int j = 0; j < cols; j++) {
            if (grid[0][j] == 1) {
                this->dfs(0, j, grid);
            }
            if (grid[rows - 1][j] == 1) {
                this->dfs(rows - 1, j, grid);
            }
        }

        int ans = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 1) {
                    ans++;
                }
            }
        }

        return ans;
    }
private:
    // 方向
    int m_dx[4] = {-1, 1, 0, 0};
    int m_dy[4] = {0, 0, -1, 1};

    void dfs(int i, int j, vector<vector<int>>& grid) {
        if (i < 0 || i >= grid.size()
            || j < 0 || j >= grid[i].size()
            || grid[i][j] == 0) {
            return;
        }

        grid[i][j] = 0;

        for (int k = 0; k < 4; k++) {
            int x = i + m_dx[k];
            int y = j + m_dy[k];

            this->dfs(x, y, grid);
        }
    }
};
// @lc code=end

