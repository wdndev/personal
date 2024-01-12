/*
 * @lc app=leetcode.cn id=463 lang=cpp
 *
 * [463] 岛屿的周长
 *
 * https://leetcode.cn/problems/island-perimeter/description/
 *
 * algorithms
 * Easy (70.06%)
 * Likes:    725
 * Dislikes: 0
 * Total Accepted:    147.4K
 * Total Submissions: 210.4K
 * Testcase Example:  '[[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]'
 *
 * 给定一个 row x col 的二维网格地图 grid ，其中：grid[i][j] = 1 表示陆地， grid[i][j] = 0 表示水域。
 * 
 * 网格中的格子 水平和垂直 方向相连（对角线方向不相连）。整个网格被水完全包围，但其中恰好有一个岛屿（或者说，一个或多个表示陆地的格子相连组成的岛屿）。
 * 
 * 岛屿中没有“湖”（“湖” 指水域在岛屿内部且不和岛屿周围的水相连）。格子是边长为 1 的正方形。网格为长方形，且宽度和高度均不超过 100
 * 。计算这个岛屿的周长。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 
 * 
 * 输入：grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
 * 输出：16
 * 解释：它的周长是上面图片中的 16 个黄色的边
 * 
 * 示例 2：
 * 
 * 
 * 输入：grid = [[1]]
 * 输出：4
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：grid = [[1,0]]
 * 输出：4
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * row == grid.length
 * col == grid[i].length
 * 1 
 * grid[i][j] 为 0 或 1
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 广度优先搜索
    int islandPerimeter(vector<vector<int>>& grid) {
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[i].size(); j++) {
                if (grid[i][j] == 1) {
                    return this->bfs(grid, i, j);
                }
            }
        }
        return -1;
    }

private:
    // 方向
    int m_dx[4] = {-1, 1, 0, 0};
    int m_dy[4] = {0, 0, -1, 1};

    int bfs(vector<vector<int>>& grid, int i, int j) {
        std::queue<std::pair<int, int>> queue;
        queue.push({i, j});
        int count = 0;
        while (!queue.empty()) {
            auto [row, col] = queue.front();
            queue.pop();
            // 避免重复计算
            grid[row][col] = 2;
            for (int k = 0; k < 4; k++) {
                int x = row + m_dx[k];
                int y = col + m_dy[k];
                // 遇到边界或水域，周长加 1
                if (x < 0 || x >= grid.size() || y < 0 || y >= grid[x].size() || grid[x][y] == 0) {
                    count++;
                // 相邻区域为陆地，则将其标记为2，加入队列
                } else if (grid[x][y] == 1) {
                    grid[x][y] = 2;
                    queue.push({x, y});
                }
            }
        }

        return count;
    }
};
// @lc code=end

