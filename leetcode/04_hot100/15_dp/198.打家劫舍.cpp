/*
 * @lc app=leetcode.cn id=198 lang=cpp
 *
 * [198] 打家劫舍
 *
 * https://leetcode.cn/problems/house-robber/description/
 *
 * algorithms
 * Medium (54.77%)
 * Likes:    2822
 * Dislikes: 0
 * Total Accepted:    853.8K
 * Total Submissions: 1.6M
 * Testcase Example:  '[1,2,3,1]'
 *
 * 
 * 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
 * 
 * 给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：[1,2,3,1]
 * 输出：4
 * 解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
 * 偷窃到的最高金额 = 1 + 3 = 4 。
 * 
 * 示例 2：
 * 
 * 
 * 输入：[2,7,9,3,1]
 * 输出：12
 * 解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
 * 偷窃到的最高金额 = 2 + 9 + 1 = 12 。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 
 * 0 
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.二维动态规划
    // a[i][0, 1] ： 增加一个维度，其中，0表示i偷，1表示i不偷
    // a[i][0] = max(a[i-1][0], a[i-1][1])：当前不偷，前一个偷还是不偷的最大值
    // a[i][1] = a[i-1][0] + nums[i]： 当前偷，等于前一个的值 + 当前的值
    int rob1(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) {
            return 0;
        }

        // 增加一个维度，其中，0表示偷，1表示不偷
        std::vector<std::vector<int>> dp(n, std::vector<int>(2, 0));

        dp[0][0] = 0;
        dp[0][1] = nums[0];

        for (int i = 1; i < n; i++) {
            // 当前不偷，前一个偷还是不偷是最大值
            dp[i][0] = std::max(dp[i - 1][0], dp[i - 1][1]);
            // 当前偷，等于上一个不偷的值 + 当前值
            dp[i][1] = dp[i - 1][0] + nums[i];
        }

        return std::max(dp[n - 1][0], dp[n - 1][1]);
    }

    // 2.一维动态规划
    // a[i] = max(a[i-1] + 0, a[i-2] + nums[i])：
    // 当前的最大值等于，上一次偷的最大值 + 0(今天不偷) 和 上上一次偷的最大值 + 偷今天
    int rob(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) {
            return 0;
        }

        if (n == 1) {
            return nums[0];
        }

        // 增加一个维度，其中，0表示偷，1表示不偷
        std::vector<int> dp(n, 0);

        dp[0] = nums[0];
        dp[1] = std::max(nums[0], nums[1]);

        int res = dp[1];

        for (int i = 2; i < n; i++){
            dp[i] = std::max(dp[i - 1] + 0, dp[i - 2] + nums[i]);

            res = std::max(res, dp[i]);
        }

        return res;
    }
};
// @lc code=end

