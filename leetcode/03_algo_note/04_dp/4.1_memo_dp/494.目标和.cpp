/*
 * @lc app=leetcode.cn id=494 lang=cpp
 *
 * [494] 目标和
 *
 * https://leetcode.cn/problems/target-sum/description/
 *
 * algorithms
 * Medium (48.42%)
 * Likes:    1833
 * Dislikes: 0
 * Total Accepted:    403.2K
 * Total Submissions: 832.6K
 * Testcase Example:  '[1,1,1,1,1]\n3'
 *
 * 给你一个非负整数数组 nums 和一个整数 target 。
 * 
 * 向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：
 * 
 * 
 * 例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
 * 
 * 
 * 返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [1,1,1,1,1], target = 3
 * 输出：5
 * 解释：一共有 5 种方法让最终目标和为 3 。
 * -1 + 1 + 1 + 1 + 1 = 3
 * +1 - 1 + 1 + 1 + 1 = 3
 * +1 + 1 - 1 + 1 + 1 = 3
 * +1 + 1 + 1 - 1 + 1 = 3
 * +1 + 1 + 1 + 1 - 1 = 3
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [1], target = 1
 * 输出：1
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= nums.length <= 20
 * 0 <= nums[i] <= 1000
 * 0 <= sum(nums[i]) <= 1000
 * -1000 <= target <= 1000
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.深度优先搜索
    int findTargetSumWays1(vector<int>& nums, int target) {
        return this->dfs(0, 0, nums, target);
    }
    int dfs(int idx, int curr_sum, std::vector<int>& nums, int target) {
        if (idx == nums.size()) {
            if (curr_sum == target) {
                return 1;
            } else {
                return 0;
            }
        }

        int ans = this->dfs(idx+1, curr_sum-nums[idx], nums, target)
                + this->dfs(idx+1, curr_sum+nums[idx], nums, target);
        return ans;
    }

    // 2.动态规划
    int findTargetSumWays(vector<int>& nums, int target) {
        int sum_nums = accumulate(nums.begin(), nums.end(), 0);
        if (abs(target) > abs(sum_nums) || (target + sum_nums) % 2 == 1) {
            return 0;
        }
        int size = (target + sum_nums) / 2;
        std::vector<int> dp(size + 1, 0);
        dp[0] = 1;

        for (auto& num : nums) {
            for (int i = size; i > num - 1; i--) {
                dp[i] = dp[i] + dp[i - num];
            }
        }

        return dp[size];
    }
        
};
// @lc code=end

