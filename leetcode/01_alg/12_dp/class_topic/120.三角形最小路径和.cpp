/*
 * @lc app=leetcode.cn id=120 lang=cpp
 *
 * [120] 三角形最小路径和
 *
 * https://leetcode.cn/problems/triangle/description/
 *
 * algorithms
 * Medium (68.61%)
 * Likes:    1278
 * Dislikes: 0
 * Total Accepted:    314.2K
 * Total Submissions: 457.9K
 * Testcase Example:  '[[2],[3,4],[6,5,7],[4,1,8,3]]'
 *
 * 给定一个三角形 triangle ，找出自顶向下的最小路径和。
 * 
 * 每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1
 * 的两个结点。也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
 * 输出：11
 * 解释：如下面简图所示：
 * ⁠  2
 * ⁠ 3 4
 * ⁠6 5 7
 * 4 1 8 3
 * 自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：triangle = [[-10]]
 * 输出：-10
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 
 * triangle[0].length == 1
 * triangle[i].length == triangle[i - 1].length + 1
 * -10^4 
 * 
 * 
 * 
 * 
 * 进阶：
 * 
 * 
 * 你可以只使用 O(n) 的额外空间（n 为三角形的总行数）来解决这个问题吗？
 * 
 * 
 */

// @lc code=start

class Solution {
public:
    // 1. DP
    //     1. 重复性（分治）：$problem(i, j) = min(sub(i+1, j) + sub(i+1, j+1)) + a(i, j)$
    //     2. 定义状态数组：$f[i, j]$
    //     3. DP方程：$f(i, j) = min(f(i+1, j), f(i + 1, j + 1)) + a[i, j]$
    int minimumTotal1(vector<vector<int>>& triangle) {
        // 注意初始化，不开新的数组也可以，直接将 triangle 当作 dp
        std::vector<std::vector<int>> dp(triangle);

        for (int i = dp.size() - 2; i >= 0; i--) {
            for (int j = dp[i].size() - 1; j >= 0; j--) {
                // 之前依据初始化过了，直接累加
                dp[i][j] += std::min(dp[i + 1][j], dp[i + 1][j + 1]);
            }
        }

        return dp[0][0];
    }
    // 2. 暴力方法：递归，n层 : left or right，2^n
    int minimumTotal(vector<vector<int>>& triangle) {
        int size = triangle.size();
        std::vector<std::vector<int>> memo(size, std::vector<int>(size, -1));

        return this->bfs(0, 0, memo, triangle);
    }

private:
    int bfs(int level, int c, std::vector<std::vector<int>>& memo, 
        std::vector<std::vector<int>>& triangle) {
        if (memo[level][c] != -1) {
            return memo[level][c];
        }

        if (level == triangle.size() - 1) {
            return memo[level][c] = triangle[level][c];
        }

        int left = this->bfs(level + 1, c, memo, triangle);
        int right = this->bfs(level + 1, c + 1, memo, triangle);

        return memo[level][c] = std::min(left, right) + triangle[level][c];
    }
};
// @lc code=end

