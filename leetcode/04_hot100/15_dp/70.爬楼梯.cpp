/*
 * @lc app=leetcode.cn id=70 lang=cpp
 *
 * [70] 爬楼梯
 *
 * https://leetcode.cn/problems/climbing-stairs/description/
 *
 * algorithms
 * Easy (54.15%)
 * Likes:    3295
 * Dislikes: 0
 * Total Accepted:    1.3M
 * Total Submissions: 2.3M
 * Testcase Example:  '2'
 *
 * 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
 * 
 * 每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：n = 2
 * 输出：2
 * 解释：有两种方法可以爬到楼顶。
 * 1. 1 阶 + 1 阶
 * 2. 2 阶
 * 
 * 示例 2：
 * 
 * 
 * 输入：n = 3
 * 输出：3
 * 解释：有三种方法可以爬到楼顶。
 * 1. 1 阶 + 1 阶 + 1 阶
 * 2. 1 阶 + 2 阶
 * 3. 2 阶 + 1 阶
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= n <= 45
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.递归， 超时
    int climbStairs1(int n) {
        if (n <= 2) {
            return n;
        }

        return this->climbStairs(n - 1) + this->climbStairs(n - 2);
    }

    // 2.递归 + 哈希cache
    int climbStairs2(int n) {
        if (n <= 2) {
            return n;
        }

        std::vector<int> memo(n+1, 0);
        return this->recur(n, memo);
    }

    int recur(int n, std::vector<int>& memo) {
        if (n <= 2) {
            return n;
        }

        if (memo[n] == 0) {
            memo[n] = this->recur(n - 1, memo) + this->recur(n - 2, memo);
        }

        return memo[n];
    }

    // 3.动态规划
    int climbStairs(int n) {
        if (n <= 2) {
            return n;
        }

        std::vector<int> dp(n + 1, 0);
        dp[0] = 0;
        dp[1] = 1;
        dp[2] = 2;

        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }

        return dp[n];
    }

};
// @lc code=end

