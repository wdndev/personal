/*
 * @lc app=leetcode.cn id=650 lang=cpp
 *
 * [650] 两个键的键盘
 *
 * https://leetcode.cn/problems/2-keys-keyboard/description/
 *
 * algorithms
 * Medium (57.79%)
 * Likes:    538
 * Dislikes: 0
 * Total Accepted:    68.8K
 * Total Submissions: 119K
 * Testcase Example:  '3'
 *
 * 最初记事本上只有一个字符 'A' 。你每次可以对这个记事本进行两种操作：
 * 
 * 
 * Copy All（复制全部）：复制这个记事本中的所有字符（不允许仅复制部分字符）。
 * Paste（粘贴）：粘贴 上一次 复制的字符。
 * 
 * 
 * 给你一个数字 n ，你需要使用最少的操作次数，在记事本上输出 恰好 n 个 'A' 。返回能够打印出 n 个 'A' 的最少操作次数。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：3
 * 输出：3
 * 解释：
 * 最初, 只有一个字符 'A'。
 * 第 1 步, 使用 Copy All 操作。
 * 第 2 步, 使用 Paste 操作来获得 'AA'。
 * 第 3 步, 使用 Paste 操作来获得 'AAA'。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：n = 1
 * 输出：0
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= n <= 1000
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    int minSteps(int n) {
        std::vector<float> dp(n+1, 0);
        for (int i = 2; i <= n; i++) {
            dp[i] = INT_MAX;
            for (int j = 1; j <= sqrt(n); j++) {
                if (i % j == 0) {
                    dp[i] = std::min(std::min(dp[i], dp[j] + i / j), dp[i / j] + j);
                }
            }
        }
        return dp[n];
    }
};
// @lc code=end

