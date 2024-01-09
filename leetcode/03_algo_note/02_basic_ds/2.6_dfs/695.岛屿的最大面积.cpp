/*
 * @lc app=leetcode.cn id=695 lang=cpp
 *
 * [695] 岛屿的最大面积
 *
 * https://leetcode.cn/problems/max-area-of-island/description/
 *
 * algorithms
 * Medium (68.09%)
 * Likes:    1044
 * Dislikes: 0
 * Total Accepted:    309.9K
 * Total Submissions: 455K
 * Testcase Example:  '[[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]'
 *
 * 给你一个大小为 m x n 的二进制矩阵 grid 。
 * 
 * 岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在 水平或者竖直的四个方向上 相邻。你可以假设 grid
 * 的四个边缘都被 0（代表水）包围着。
 * 
 * 岛屿的面积是岛上值为 1 的单元格的数目。
 * 
 * 计算并返回 grid 中最大的岛屿面积。如果没有岛屿，则返回面积为 0 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：grid =
 * [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
 * 输出：6
 * 解释：答案不应该是 11 ，因为岛屿只能包含水平或垂直这四个方向上的 1 。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：grid = [[0,0,0,0,0,0,0,0]]
 * 输出：0
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * m == grid.length
 * n == grid[i].length
 * 1 <= m, n <= 50
 * grid[i][j] 为 0 或 1
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 填海造陆
    // 遍历地图，遇到一个1，将1进行BFS/DFS遍历，将周围的1全部变为0，记录最大岛屿面积
    // 去最大岛屿面积
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        int max_area = 0;
        m_grid = grid;

        for (int i = 0; i < m_grid.size(); i++) {
            for (int j = 0; j < m_grid[i].size(); j++) {
                if (m_grid[i][j] == 0) {
                    continue;
                }
                max_area = std::max(max_area, this->sink(i, j));
            }
        }

        return max_area;
    }
private:
    // 方向
    int m_dx[4] = {-1, 1, 0, 0};
    int m_dy[4] = {0, 0, -1, 1};
    // 全局地图
    std::vector<std::vector<int>> m_grid;

    // bfs遍历
    int sink(int i, int j) {
        if (m_grid[i][j] == 0) {
            return 0;
        }
        int area = 1;
        // 将i，j变为0
        m_grid[i][j] = 0;
        // 开始遍历四个方向
        for (int k = 0; k < 4; k++) {
            int x = i + m_dx[k];
            int y = j + m_dy[k];

            if (x >=0 && x < m_grid.size() && y >= 0 && y < m_grid[i].size()) {
                // 遇到0跳过
                // 遇到1开始递归
                if (m_grid[x][y] == 0) {
                    continue;
                }
                area += this->sink(x, y);
            }
        }
        return area;
    }
};
// @lc code=end

