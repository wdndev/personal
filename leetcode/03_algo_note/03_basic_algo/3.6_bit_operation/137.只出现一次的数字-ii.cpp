/*
 * @lc app=leetcode.cn id=137 lang=cpp
 *
 * [137] 只出现一次的数字 II
 *
 * https://leetcode.cn/problems/single-number-ii/description/
 *
 * algorithms
 * Medium (72.38%)
 * Likes:    1176
 * Dislikes: 0
 * Total Accepted:    187.7K
 * Total Submissions: 259.4K
 * Testcase Example:  '[2,2,3,2]'
 *
 * 给你一个整数数组 nums ，除某个元素仅出现 一次 外，其余每个元素都恰出现 三次 。请你找出并返回那个只出现了一次的元素。
 * 
 * 你必须设计并实现线性时间复杂度的算法且使用常数级空间来解决此问题。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [2,2,3,2]
 * 输出：3
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [0,1,0,1,0,1,99]
 * 输出：99
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= nums.length <= 3 * 10^4
 * -2^31 <= nums[i] <= 2^31 - 1
 * nums 中，除某个元素仅出现 一次 外，其余每个元素都恰出现 三次
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.哈希表，利用哈希表统计出每个元素出现的次数
    int singleNumber(vector<int>& nums) {
        std::unordered_map<int, int> freq;
        for (int num : nums)
        {
            freq[num]++;
        }
        int ans = 0;
        for (auto [num, time] : freq)
        {
            if (time == 1)
            {
                ans = num;
                break;
            }
        }

        return ans;
    }
    // 2.
};
// @lc code=end

