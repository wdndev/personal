/*
 * @lc app=leetcode.cn id=1004 lang=cpp
 *
 * [1004] 最大连续1的个数 III
 *
 * https://leetcode.cn/problems/max-consecutive-ones-iii/description/
 *
 * algorithms
 * Medium (59.30%)
 * Likes:    670
 * Dislikes: 0
 * Total Accepted:    146.5K
 * Total Submissions: 247.1K
 * Testcase Example:  '[1,1,1,0,0,0,1,1,1,1,0]\n2'
 *
 * 给定一个二进制数组 nums 和一个整数 k，如果可以翻转最多 k 个 0 ，则返回 数组中连续 1 的最大个数 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [1,1,1,0,0,0,1,1,1,1,0], K = 2
 * 输出：6
 * 解释：[1,1,1,0,0,1,1,1,1,1,1]
 * 粗体数字从 0 翻转到 1，最长的子数组长度为 6。
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], K = 3
 * 输出：10
 * 解释：[0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1]
 * 粗体数字从 0 翻转到 1，最长的子数组长度为 10。
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= nums.length <= 10^5
 * nums[i] 不是 0 就是 1
 * 0 <= k <= nums.length
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    int longestOnes(vector<int>& nums, int k) {
        int max_count = 0;
        int zero_count = 0;
        int left = 0;
        int right = 0;

        while (right < nums.size()) {
            // 统计0的个数
            if (nums[right] == 0) {
                zero_count++;
            } 
            right++;
            // 如果0的个数超过k时，将left开始右移，缩小滑动窗口范围，并减小0元素个数
            // 同时维护 max_count
            if (zero_count > k) {
               if (nums[left] == 0) {
                    zero_count--;
               }
               left++;
            }
            max_count = std::max(max_count, right - left);
        }

        return max_count;
    }
};
// @lc code=end

