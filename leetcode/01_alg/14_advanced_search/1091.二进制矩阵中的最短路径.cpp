/*
 * @lc app=leetcode.cn id=1091 lang=cpp
 *
 * [1091] 二进制矩阵中的最短路径
 *
 * https://leetcode.cn/problems/shortest-path-in-binary-matrix/description/
 *
 * algorithms
 * Medium (40.41%)
 * Likes:    349
 * Dislikes: 0
 * Total Accepted:    79.9K
 * Total Submissions: 197.8K
 * Testcase Example:  '[[0,1],[1,0]]'
 *
 * 给你一个 n x n 的二进制矩阵 grid 中，返回矩阵中最短 畅通路径 的长度。如果不存在这样的路径，返回 -1 。
 * 
 * 二进制矩阵中的 畅通路径 是一条从 左上角 单元格（即，(0, 0)）到 右下角 单元格（即，(n - 1, n -
 * 1)）的路径，该路径同时满足下述要求：
 * 
 * 
 * 路径途经的所有单元格的值都是 0 。
 * 路径中所有相邻的单元格应当在 8 个方向之一 上连通（即，相邻两单元之间彼此不同且共享一条边或者一个角）。
 * 
 * 
 * 畅通路径的长度 是该路径途经的单元格总数。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：grid = [[0,1],[1,0]]
 * 输出：2
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：grid = [[0,0,0],[1,1,0],[1,1,0]]
 * 输出：4
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：grid = [[1,0,0],[1,1,0],[1,1,0]]
 * 输出：-1
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * n == grid.length
 * n == grid[i].length
 * 1 <= n <= 100
 * grid[i][j] 为 0 或 1
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // bfs,超时
    int shortestPathBinaryMatrix(vector<vector<int>>& grid) {
        int n = grid.size();
        if (grid[0][0] == 1 || grid[n - 1][n - 1] == 1) {
            return -1;
        } else if (n <= 2) {
            return n;
        }

        // std::vector<int>  [row, col, distance]
        std::queue<std::vector<int>> queue;
        queue.push({0, 0, 2});

        int dx[] = {-1, -1, -1, 0, 0, 1, 1, 1};
        int dy[] = {-1, 0, 1, -1, 1, -1, 0, 1};

        while (!queue.empty())
        {
            std::vector<int> row_col_d = queue.front();
            queue.pop();
            int row = row_col_d[0];
            int col = row_col_d[1];
            int d = row_col_d[2];
            for (int k = 0; k < 8; k++) {
                int x = row + dx[k];
                int y = col + dy[k];
                
                if (x >= 0 && x < n && y >= 0 && y < n && grid[x][y] == 0) {
                    if (x == n - 1 && y == n - 1) {
                        return d;
                    }
                    queue.push({x, y, d+1});
                    grid[x][y] == 1;
                }
            }
        }
        
        return -1;
    }
};
// @lc code=end

