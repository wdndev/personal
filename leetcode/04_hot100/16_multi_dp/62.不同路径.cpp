// @before-stub-for-debug-begin
#include <vector>
#include <string>
#include "commoncppproblem62.h"

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=62 lang=cpp
 *
 * [62] 不同路径
 *
 * https://leetcode.cn/problems/unique-paths/description/
 *
 * algorithms
 * Medium (67.89%)
 * Likes:    1950
 * Dislikes: 0
 * Total Accepted:    704.4K
 * Total Submissions: 1M
 * Testcase Example:  '3\n7'
 *
 * 一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
 * 
 * 机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
 * 
 * 问总共有多少条不同的路径？
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：m = 3, n = 7
 * 输出：28
 * 
 * 示例 2：
 * 
 * 
 * 输入：m = 3, n = 2
 * 输出：3
 * 解释：
 * 从左上角开始，总共有 3 条路径可以到达右下角。
 * 1. 向右 -> 向下 -> 向下
 * 2. 向下 -> 向下 -> 向右
 * 3. 向下 -> 向右 -> 向下
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：m = 7, n = 3
 * 输出：28
 * 
 * 
 * 示例 4：
 * 
 * 
 * 输入：m = 3, n = 3
 * 输出：6
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= m, n <= 100
 * 题目数据保证答案小于等于 2 * 10^9
 * 
 * 
 */

// @lc code=start
// 状态方程：f(x,y) = f(x-1,y) + f(x, y-1)
class Solution {
public:
    // 1.递归
    int uniquePaths1(int m, int n) {
        return this->dfs(m, n);
    }
    int dfs(int x, int y) {
        if (x <= 0 || y <= 0) {
            return 0;
        }
        if (x == 1 && y == 1) {
            return 1;
        }
        return this->dfs(x - 1, y) + this->dfs(x, y - 1);
    }

    // 2.递归 + 记忆化搜索
    int uniquePaths2(int m, int n) {
        std::vector<std::vector<int>> memo(m + 1, std::vector<int>(n + 1, -1));

        return this->dfs_memo(m, n, memo);
    }
    int dfs_memo(int x, int y, std::vector<std::vector<int>>& memo) {
        if (x <= 0 || y <= 0) {
            return 0;
        }
        if (x == 1 && y == 1) {
            return 1;
        }
        
        if (memo[x][y] == -1) {
            memo[x][y] = this->dfs_memo(x - 1, y, memo) + this->dfs_memo(x, y - 1, memo);
        }

        return memo[x][y];
    }

    // 3.动态规划
    // int uniquePaths(int m, int n) {

    //     std::vector<std::vector<long long>> dp(m, std::vector<long long>(n, 0));

    //     dp[m - 1][n - 1] = 1;

    //     // 处理最后一列
    //     for (int i = m - 2; i >= 0; i--) {
    //         dp[i][n - 1] = 1;
    //     }

    //     // 处理最后一行
    //     for (int j = n - 2; j >= 0; j--) {
    //         if (dp[m - 1][j + 1] == 0) {
    //             dp[m - 1][j] = 0;
    //         } else {
    //             dp[m - 1][j] = 1;
    //         }
    //     }

    //     for (int i = m - 2; i >= 0; i--) {
    //         for (int j = n - 2; j >= 0; j--) {
    //             dp[i][j] = dp[i + 1][j] + dp[i][j + 1];
    //         }
    //     }

    //     return dp[0][0];
    // }

    int uniquePaths(int m, int n) {
        std::vector<std::vector<int>> dp(m+1,std::vector<int>(n+1, 0));
        dp[0][1] = 1;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m][n];
    }
};
// @lc code=end

