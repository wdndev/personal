/*
 * @lc app=leetcode.cn id=209 lang=cpp
 *
 * [209] 长度最小的子数组
 *
 * https://leetcode.cn/problems/minimum-size-subarray-sum/description/
 *
 * algorithms
 * Medium (46.45%)
 * Likes:    2011
 * Dislikes: 0
 * Total Accepted:    666.6K
 * Total Submissions: 1.4M
 * Testcase Example:  '7\n[2,3,1,2,4,3]'
 *
 * 给定一个含有 n 个正整数的数组和一个正整数 target 。
 * 
 * 找出该数组中满足其总和大于等于 target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr]
 * ，并返回其长度。如果不存在符合条件的子数组，返回 0 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：target = 7, nums = [2,3,1,2,4,3]
 * 输出：2
 * 解释：子数组 [4,3] 是该条件下的长度最小的子数组。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：target = 4, nums = [1,4,4]
 * 输出：1
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：target = 11, nums = [1,1,1,1,1,1,1,1]
 * 输出：0
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= target <= 10^9
 * 1 <= nums.length <= 10^5
 * 1 <= nums[i] <= 10^5
 * 
 * 
 * 
 * 
 * 进阶：
 * 
 * 
 * 如果你已经实现 O(n) 时间复杂度的解法, 请尝试设计一个 O(n log(n)) 时间复杂度的解法。
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 滑动窗口
    int minSubArrayLen(int target, vector<int>& nums) {
        if (nums.size() == 0) {
            return 0;
        }
        int left = 0;
        int right = 0;
        int window_sum = 0;
        int ans = nums.size() + 1;

        while (right < nums.size()) {
            window_sum += nums[right];
            // 如果连续数组之和超过target
            // 更新ans为right - left + 1
            // 从sum中移除 nums[left]
            // left后移一位
            while (window_sum >= target) {
                ans = std::min(ans, right - left + 1);
                window_sum -= nums[left];
                left++;
            } 
            right++;
        }

        if (ans == nums.size() + 1) {
            ans = 0;
        }

        return ans == nums.size() + 1 ? 0 : ans;
    }

    // 暴力解法，两层循环, 超出时间限制
    int minSubArrayLen_nonono(int target, vector<int>& nums) {
        int n = nums.size();
        if (n == 0) {
            return 0;
        }

        int ans = INT_MAX;
        // 对于每个开始下标i，需要找到大于或等于i的最小下标j
        // 使得 nums[i]到nums[j]的元素之和大于或等于s
        // 最后更新数组的最小长度，j - i + 1
        for (int i = 0; i < n; i++) {
            int sum = 0;
            for (int j = i; j < n; j++) {
                sum += nums[j];
                if (sum >= target) {
                    ans = std::min(ans, j - i + 1);
                    break;
                }
            }
        }
        return ans == INT_MAX ? 0 : ans;
    }
};
// @lc code=end

