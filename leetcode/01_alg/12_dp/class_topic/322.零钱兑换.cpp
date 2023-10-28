/*
 * @lc app=leetcode.cn id=322 lang=cpp
 *
 * [322] 零钱兑换
 *
 * https://leetcode.cn/problems/coin-change/description/
 *
 * algorithms
 * Medium (46.90%)
 * Likes:    2607
 * Dislikes: 0
 * Total Accepted:    697.6K
 * Total Submissions: 1.5M
 * Testcase Example:  '[1,2,5]\n11'
 *
 * 给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。
 * 
 * 计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。
 * 
 * 你可以认为每种硬币的数量是无限的。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：coins = [1, 2, 5], amount = 11
 * 输出：3 
 * 解释：11 = 5 + 5 + 1
 * 
 * 示例 2：
 * 
 * 
 * 输入：coins = [2], amount = 3
 * 输出：-1
 * 
 * 示例 3：
 * 
 * 
 * 输入：coins = [1], amount = 0
 * 输出：0
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= coins.length <= 12
 * 1 <= coins[i] <= 2^31 - 1
 * 0 <= amount <= 10^4
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.动态规划
    int coinChange1(vector<int>& coins, int amount) {
        int max_amount = amount + 1;
        std::vector<int> dp(amount + 1, max_amount);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int j = 0; j < coins.size(); j++) {
                if (coins[j] <= i) {
                    dp[i] = std::min(dp[i], dp[i - coins[j]] + 1);
                }
            }
        }

        return dp[amount] > amount ? -1 : dp[amount];
    }

    // 2.递归方法
    int coinChange(vector<int>& coins, int amount) {
        if (amount < 1) {
            return 0;
        }

        std::vector<int> count(amount, 0);
        return this->dp(coins, amount, count);
    }

private:
    // rem : 剩余的价值
    // count[rem] 最小到达的数量
    int dp(std::vector<int>& coins, int rem, std::vector<int>& count) {
        // 无效数据
        if (rem < 0) {
            return -1;
        }
        // 终止条件
        if (rem == 0) {
            return 0;
        }
        if (count[rem - 1] != 0) {
            return count[rem - 1];
        }
        int min = INT_MAX;
        for (int& coin : coins) {
            int res = this->dp(coins, rem - coin, count);
            if (res >= 0 && res < min) {
                min = res + 1;
            }
        }
        count[rem - 1] = (min == INT_MAX) ? -1 : min;
        return count[rem - 1];
    }
};
// @lc code=end

