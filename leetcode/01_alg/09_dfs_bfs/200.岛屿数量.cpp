/*
 * @lc app=leetcode.cn id=200 lang=cpp
 *
 * [200] 岛屿数量
 *
 * https://leetcode.cn/problems/number-of-islands/description/
 *
 * algorithms
 * Medium (59.68%)
 * Likes:    2336
 * Dislikes: 0
 * Total Accepted:    720.8K
 * Total Submissions: 1.2M
 * Testcase Example:  '[["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]'
 *
 * 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
 * 
 * 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
 * 
 * 此外，你可以假设该网格的四条边均被水包围。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：grid = [
 * ⁠ ["1","1","1","1","0"],
 * ⁠ ["1","1","0","1","0"],
 * ⁠ ["1","1","0","0","0"],
 * ⁠ ["0","0","0","0","0"]
 * ]
 * 输出：1
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：grid = [
 * ⁠ ["1","1","0","0","0"],
 * ⁠ ["1","1","0","0","0"],
 * ⁠ ["0","0","1","0","0"],
 * ⁠ ["0","0","0","1","1"]
 * ]
 * 输出：3
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * m == grid.length
 * n == grid[i].length
 * 1 
 * grid[i][j] 的值为 '0' 或 '1'
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.洪水算法(填海造陆)
    //   遍历地图，遇到一个1，将1进行BFS/DFS遍历，将周围的1全部变为0，操作一次记为1
    //   操作次数，就是岛屿的数量
    int numIslands(vector<vector<char>>& grid) {
        int island_num = 0;
        m_grid = grid;

        for (int i = 0; i < m_grid.size(); i++) {
            for (int j = 0; j < m_grid[i].size(); j++) {
                if (m_grid[i][j] == '0') {
                    continue;
                }
                // if (this->sink(i, j)) {
                //     island_num++;
                // }
                island_num += this->sink(i, j);
            }
        }

        return island_num;
    }
private:
    int sink(int i, int j) {
        if (m_grid[i][j] == '0') {
            return 0;
        }

        // i, j = "1"
        m_grid[i][j] = '0';

        for (int k = 0; k < 4; k++) {
            int x = i + m_dx[k];
            int y = j + m_dy[k];

            if (x >= 0 && x < m_grid.size() && y >= 0 && y < m_grid[i].size()) {
                if (m_grid[x][y] == '0') {
                    continue;
                }
                this->sink(x, y);
            }
        }

        return 1;
    }

    // 方向
    int m_dx[4] = {-1, 1, 0, 0};
    int m_dy[4] = {0, 0, -1, 1};

    std::vector<std::vector<char>> m_grid;
};
// @lc code=end

