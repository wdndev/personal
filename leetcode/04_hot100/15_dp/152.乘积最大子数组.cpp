/*
 * @lc app=leetcode.cn id=152 lang=cpp
 *
 * [152] 乘积最大子数组
 *
 * https://leetcode.cn/problems/maximum-product-subarray/description/
 *
 * algorithms
 * Medium (43.14%)
 * Likes:    2143
 * Dislikes: 0
 * Total Accepted:    393.5K
 * Total Submissions: 912.2K
 * Testcase Example:  '[2,3,-2,4]'
 *
 * 给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。
 * 
 * 测试用例的答案是一个 32-位 整数。
 * 
 * 子数组 是数组的连续子序列。
 * 
 * 
 * 
 * 示例 1:
 * 
 * 
 * 输入: nums = [2,3,-2,4]
 * 输出: 6
 * 解释: 子数组 [2,3] 有最大乘积 6。
 * 
 * 
 * 示例 2:
 * 
 * 
 * 输入: nums = [-2,0,-1]
 * 输出: 0
 * 解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。
 * 
 * 
 * 
 * 提示:
 * 
 * 
 * 1 <= nums.length <= 2 * 10^4
 * -10 <= nums[i] <= 10
 * nums 的任何前缀或后缀的乘积都 保证 是一个 32-位 整数
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int max_value = nums[0];
        int min_value = nums[0];
        int ans = nums[0];

        for (int i = 1; i < nums.size(); i++) {
            // 由于存在负数，那么会导致最大的变最小，最小的变最大。
            if (nums[i] < 0) {
                int tmp = max_value;
                max_value = min_value;
                min_value = tmp;
            }

            max_value = std::max(max_value * nums[i], nums[i]);
            min_value = std::min(min_value * nums[i], nums[i]);

            ans = std::max(ans, max_value);
        }

        return ans;
    }
};
// @lc code=end

