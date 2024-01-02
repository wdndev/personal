/*
 * @lc app=leetcode.cn id=674 lang=cpp
 *
 * [674] 最长连续递增序列
 *
 * https://leetcode.cn/problems/longest-continuous-increasing-subsequence/description/
 *
 * algorithms
 * Easy (56.36%)
 * Likes:    437
 * Dislikes: 0
 * Total Accepted:    222.8K
 * Total Submissions: 394.7K
 * Testcase Example:  '[1,3,5,4,7]'
 *
 * 给定一个未经排序的整数数组，找到最长且 连续递增的子序列，并返回该序列的长度。
 * 
 * 连续递增的子序列 可以由两个下标 l 和 r（l < r）确定，如果对于每个 l ，都有 nums[i] < nums[i + 1] ，那么子序列
 * [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] 就是连续递增子序列。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [1,3,5,4,7]
 * 输出：3
 * 解释：最长连续递增序列是 [1,3,5], 长度为3。
 * 尽管 [1,3,5,7] 也是升序的子序列, 但它不是连续的，因为 5 和 7 在原数组里被 4 隔开。 
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [2,2,2,2,2]
 * 输出：1
 * 解释：最长连续递增序列是 [2], 长度为1。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 
 * -10^9 
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1、动态规划
    int findLengthOfLCIS1(vector<int>& nums) {
        int size = nums.size();
        // 注意，初始化为1，而不是0
        std::vector<int> dp(size, 1);
        int max_len = 1;
        for (int i = 1; i < size; i++) {
            // 不连续递增子序列
            if (nums[i] > nums[i - 1]) {
                dp[i] = dp[i - 1] + 1;
                max_len = std::max(dp[i], max_len);
            }
        }
        return max_len;
    }

    // 2.滑动窗口
    int findLengthOfLCIS(vector<int>& nums) {
        int size = nums.size();
        int left = 0;
        int right = 0;
        int window_len = 0;
        int max_len = 0;

        while (right < size) {
            window_len++;

            // 如果不满足连续递增序列，将left移动到窗口最右侧，重置当前窗口长度
            if (right > 0 && nums[right - 1] >= nums[right]) {
                left = right;
                window_len = 1;
            }
            max_len = std::max(max_len, window_len);
            right++;
        }

        return max_len;
    }
};
// @lc code=end

