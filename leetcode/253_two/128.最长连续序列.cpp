/*
 * @lc app=leetcode.cn id=128 lang=cpp
 *
 * [128] 最长连续序列
 *
 * https://leetcode.cn/problems/longest-consecutive-sequence/description/
 *
 * algorithms
 * Medium (51.80%)
 * Likes:    2077
 * Dislikes: 0
 * Total Accepted:    632.9K
 * Total Submissions: 1.2M
 * Testcase Example:  '[100,4,200,1,3,2]'
 *
 * 给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
 * 
 * 请你设计并实现时间复杂度为 O(n) 的算法解决此问题。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [100,4,200,1,3,2]
 * 输出：4
 * 解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [0,3,7,2,5,8,4,6,0,1]
 * 输出：9
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 0 
 * -10^9 
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.排序+遍历
    // 和上一项差值为1或0，则连续，否则不连续
    int longestConsecutive(vector<int>& nums) {
        int max_len = 0;
        if (nums.size() == 0) {
            return 0;
        } else if (nums.size() == 1) {
            return 1;
        }

        // 排序
        std::sort(nums.begin(), nums.end());
        int tmp = 0;
        for (int i = 1; i < nums.size(); i++) {
            int diff = nums[i] - nums[i - 1];
            // diff == 0, 有相同的数字，跳过，连续值不变
            if (diff == 0) {
                continue;
            } else if (diff == 1) {
                // diff=1，连续，连续值加1
                tmp += 1;
            } else {
                // 其他不是连续，将连续值置为 0 
                tmp = 0;
            }

            max_len = std::max(max_len, tmp);
        }

        // 因为从第 2 个数组开始遍历的
        return max_len + 1;
    }
};
// @lc code=end

