/*
 * @lc app=leetcode.cn id=343 lang=cpp
 *
 * [343] 整数拆分
 *
 * https://leetcode.cn/problems/integer-break/description/
 *
 * algorithms
 * Medium (62.75%)
 * Likes:    1334
 * Dislikes: 0
 * Total Accepted:    301.5K
 * Total Submissions: 480.2K
 * Testcase Example:  '2'
 *
 * 给定一个正整数 n ，将其拆分为 k 个 正整数 的和（ k >= 2 ），并使这些整数的乘积最大化。
 * 
 * 返回 你可以获得的最大乘积 。
 * 
 * 
 * 
 * 示例 1:
 * 
 * 
 * 输入: n = 2
 * 输出: 1
 * 解释: 2 = 1 + 1, 1 × 1 = 1。
 * 
 * 示例 2:
 * 
 * 
 * 输入: n = 10
 * 输出: 36
 * 解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36。
 * 
 * 
 * 
 * 提示:
 * 
 * 
 * 2 <= n <= 58
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    int integerBreak(int n) {
        std::vector<int> dp(n+1, 0);
        for (int i = 2; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                dp[i] = std::max(std::max(dp[i], (i-j)*j), dp[i-j]*j);
            }
        }
        return dp[n];
    }
};
// @lc code=end

