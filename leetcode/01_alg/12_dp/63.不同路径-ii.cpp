// @before-stub-for-debug-begin
#include <vector>
#include <string>
#include "commoncppproblem63.h"

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=63 lang=cpp
 *
 * [63] 不同路径 II
 *
 * https://leetcode.cn/problems/unique-paths-ii/description/
 *
 * algorithms
 * Medium (41.17%)
 * Likes:    1168
 * Dislikes: 0
 * Total Accepted:    420.6K
 * Total Submissions: 1M
 * Testcase Example:  '[[0,0,0],[0,1,0],[0,0,0]]'
 *
 * 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
 * 
 * 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish”）。
 * 
 * 现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？
 * 
 * 网格中的障碍物和空位置分别用 1 和 0 来表示。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
 * 输出：2
 * 解释：3x3 网格的正中间有一个障碍物。
 * 从左上角到右下角一共有 2 条不同的路径：
 * 1. 向右 -> 向右 -> 向下 -> 向下
 * 2. 向下 -> 向下 -> 向右 -> 向右
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：obstacleGrid = [[0,1],[0,0]]
 * 输出：1
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * m == obstacleGrid.length
 * n == obstacleGrid[i].length
 * 1 <= m, n <= 100
 * obstacleGrid[i][j] 为 0 或 1
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.递归
    int uniquePathsWithObstacles1(vector<vector<int>>& obstacleGrid) {
        int m = obstacleGrid.size();
        int n = obstacleGrid[0].size();
        std::vector<int> cache(n * m, -1);

        return this->count_path(cache, obstacleGrid, 0, 0);
    }

    // 2.动态规划
    int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
        int row = obstacleGrid.size();
        int col = obstacleGrid[0].size();

        std::vector<std::vector<long long>> dp(row, std::vector<long long>(col, 0));

        // 判断终点是否有障碍物
        dp[row - 1][col - 1] = (obstacleGrid[row - 1][col - 1] == 1) ? 0 : 1;

        // 处理最后一列
        for (int i = row - 2; i >= 0; i--) {
            if (obstacleGrid[i][col - 1] == 1 || dp[i + 1][col - 1] == 0) {
                dp[i][col - 1] = 0;
            } else {
                dp[i][col - 1] = 1;
            }
        }

        // 处理最后一行
        for (int j = col - 2; j >= 0; j--) {
            if (obstacleGrid[row - 1][j] == 1 || dp[row - 1][j + 1] == 0) {
                dp[row - 1][j] = 0;
            } else {
                dp[row - 1][j] = 1;
            }
        }

        for (int i = row - 2; i >= 0; i--) {
            for (int j = col - 2; j >= 0; j--) {
                // 如果当前格子是障碍物
                if (obstacleGrid[i][j] == 1) {
                    dp[i][j] = 0;
                } else {
                    // 路径总和: opt[i, j] = opt[i + 1, j] + opt[i, j+1]
                    dp[i][j] = dp[i + 1][j] + dp[i][j + 1];
                }
            }
        }

        return dp[0][0];
    }


private:
    int count_path(std::vector<int>& cache, 
            std::vector<std::vector<int>>& grid, int row, int col) {
        // 边界
        if (row >= grid.size() || col >= grid[0].size()) {
            return 0;
        }

        // 障碍
        if (grid[row][col] == 1) {
            return 0;
        }

        // 到达终点
        if (row == grid.size() - 1 && col == grid[0].size() - 1) {
            return 1;
        }

        // 继续往右、往下递归调用
        if (cache[col * grid.size() + row] != -1) {
            return cache[col * grid.size() + row];
        } else {
            int res = this->count_path(cache, grid, row + 1, col) + this->count_path(cache, grid, row, col + 1);
            cache[col * grid.size() + row] = res;
            return res;
        }  
    }
};
// @lc code=end

