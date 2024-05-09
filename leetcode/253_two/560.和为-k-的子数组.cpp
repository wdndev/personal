/*
 * @lc app=leetcode.cn id=560 lang=cpp
 *
 * [560] 和为 K 的子数组
 *
 * https://leetcode.cn/problems/subarray-sum-equals-k/description/
 *
 * algorithms
 * Medium (44.07%)
 * Likes:    2343
 * Dislikes: 0
 * Total Accepted:    457.3K
 * Total Submissions: 1M
 * Testcase Example:  '[1,1,1]\n2'
 *
 * 给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的子数组的个数 。
 * 
 * 子数组是数组中元素的连续非空序列。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [1,1,1], k = 2
 * 输出：2
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [1,2,3], k = 3
 * 输出：2
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= nums.length <= 2 * 10^4
 * -1000 <= nums[i] <= 1000
 * -10^7 <= k <= 10^7
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.前缀和，
    // 通过计算前缀和，将问题转化为求解两个前缀和之差等于k的情况
    // 前缀和 prefix_sum[i] : 从数组起始位置到第i个位置的元素之和；
    // 遍历数组，计算前缀和，并使用一个哈希表存储每个前缀和出现的次数
    int subarraySum(vector<int>& nums, int k) {
        int count = 0;
        // prefix_sum[i] 表示从数组起始位置到第i个位置的元素之和
        int prefix_sum = 0;
        // hash表
        std::unordered_map<int, int> map;
        // 初始化前缀和为0的次数为1
        map[0] = 1;

        for (auto& x : nums) {
            prefix_sum += x;
            // 检查是否存在 prefix_sum[j] - k的前缀和
            // 如果存在，说明从某个位置到当前位置的连续子数组的和为k
            if (map.find(prefix_sum - k) != map.end()) {
                count += map[prefix_sum - k];
            }
            map[prefix_sum]++;
        }

        return count;
    }
};
// @lc code=end

