/*
 * @lc app=leetcode.cn id=994 lang=cpp
 *
 * [994] 腐烂的橘子
 *
 * https://leetcode.cn/problems/rotting-oranges/description/
 *
 * algorithms
 * Medium (51.34%)
 * Likes:    873
 * Dislikes: 0
 * Total Accepted:    181.4K
 * Total Submissions: 348.6K
 * Testcase Example:  '[[2,1,1],[1,1,0],[0,1,1]]'
 *
 * 在给定的 m x n 网格 grid 中，每个单元格可以有以下三个值之一：
 * 
 * 
 * 值 0 代表空单元格；
 * 值 1 代表新鲜橘子；
 * 值 2 代表腐烂的橘子。
 * 
 * 
 * 每分钟，腐烂的橘子 周围 4 个方向上相邻 的新鲜橘子都会腐烂。
 * 
 * 返回 直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 
 * 
 * 输入：grid = [[2,1,1],[1,1,0],[0,1,1]]
 * 输出：4
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：grid = [[2,1,1],[0,1,1],[1,0,1]]
 * 输出：-1
 * 解释：左下角的橘子（第 2 行， 第 0 列）永远不会腐烂，因为腐烂只会发生在 4 个方向上。
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：grid = [[0,2]]
 * 输出：0
 * 解释：因为 0 分钟时已经没有新鲜橘子了，所以答案就是 0 。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * m == grid.length
 * n == grid[i].length
 * 1 <= m, n <= 10
 * grid[i][j] 仅为 0、1 或 2
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.刚开始将所有腐烂的橘子位置压入队列，并统计新鲜橘子的数量
    // 2.bfs从烂橘子开始遍历，然所有新鲜橘子腐烂，并把本次腐烂的橘子压入队列
    // 3.腐烂的橘子向其四周扩散，层层迭代
    // 4.最后判断时间
    int orangesRotting(vector<vector<int>>& grid) {
        int row = grid.size();
        int col = grid[0].size();

        // 新鲜橘子数量
        int flash_cnt = 0;
        // 步数
        int step = 0;

        std::queue<std::pair<int, int>> que;

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (grid[i][j] == 1) {
                    flash_cnt++;
                } else if (grid[i][j] == 2) {
                    que.push({i, j});
                }
            }
        }

        int dx[4] = {-1, 1, 0, 0};
        int dy[4] = {0, 0, -1, 1};
        while (flash_cnt > 0 && !que.empty()) {
            step++;
            
            int size = que.size();

            // 所有腐烂橘子感染一次
            for (int s = 0; s < size; s++) {
                // 取出烂橘子
                std::pair<int, int> coord = que.front();
                que.pop();
                // 四个方向
                for (int k = 0; k < 4; k++) {
                    int x = coord.first + dx[k];
                    int y = coord.second + dy[k];
                    // x,y 不越界，并且只感染新鲜橘子
                    if (x >= 0 && x < row && y >= 0 && y < col && grid[x][y] == 1) {
                        grid[x][y] = 2;

                        // 感染的橘子，加入队列
                        que.push({x, y});

                        // 新鲜橘子 -1
                        flash_cnt--;
                    }
                }
            }
        }

        // 如果bfs结束，还有橘子
        if (flash_cnt > 0) {
            return -1;
        } else {
            return step;
        }

    }
};
// @lc code=end

