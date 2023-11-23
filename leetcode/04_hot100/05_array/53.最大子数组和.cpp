/*
 * @lc app=leetcode.cn id=53 lang=cpp
 *
 * [53] 最大子数组和
 *
 * https://leetcode.cn/problems/maximum-subarray/description/
 *
 * algorithms
 * Medium (54.94%)
 * Likes:    6387
 * Dislikes: 0
 * Total Accepted:    1.5M
 * Total Submissions: 2.8M
 * Testcase Example:  '[-2,1,-3,4,-1,2,1,-5,4]'
 *
 * 给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
 * 
 * 子数组 是数组中的一个连续部分。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
 * 输出：6
 * 解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [1]
 * 输出：1
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：nums = [5,4,-1,7,8]
 * 输出：23
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= nums.length <= 10^5
 * -10^4 <= nums[i] <= 10^4
 * 
 * 
 * 
 * 
 * 进阶：如果你已经实现复杂度为 O(n) 的解法，尝试使用更为精妙的 分治法 求解。
 * 
 */

// @lc code=start
class Solution {
public:
    // 1. DP求解：
    //     1. 分治（子问题）：如果第i个元素，则子序列和是多少？$max_sum(i) = Max(max_sum(i-1), 0) + a[i]$
    //     2. 状态数组定义：$f[1]$
    //     3. DP方程：$f(i) = Max(f(i-1), 0) + a[i]$
    int maxSubArray(vector<int>& nums) {
        std::vector<int> dp(nums);
        int max_sum = dp[0];

        for (int i = 1; i < nums.size(); i++) {
            dp[i] = std::max(dp[i], dp[i] + dp[i - 1]);
            max_sum = dp[i] >= max_sum ? dp[i] : max_sum;
        }

        return max_sum;
    }
};
// @lc code=end

