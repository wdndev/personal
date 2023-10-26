/*
 * @lc app=leetcode.cn id=169 lang=cpp
 *
 * [169] 多数元素
 *
 * https://leetcode.cn/problems/majority-element/description/
 *
 * algorithms
 * Easy (66.53%)
 * Likes:    2014
 * Dislikes: 0
 * Total Accepted:    788.6K
 * Total Submissions: 1.2M
 * Testcase Example:  '[3,2,3]'
 *
 * 给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。
 * 
 * 你可以假设数组是非空的，并且给定的数组总是存在多数元素。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [3,2,3]
 * 输出：3
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [2,2,1,1,1,2,2]
 * 输出：2
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * n == nums.length
 * 1 <= n <= 5 * 10^4
 * -10^9 <= nums[i] <= 10^9
 * 
 * 
 * 
 * 
 * 进阶：尝试设计时间复杂度为 O(n)、空间复杂度为 O(1) 的算法解决此问题。
 * 
 */

// @lc code=start
class Solution {
public:
    // 使用哈希映射（HashMap）来存储每个元素以及出现的次数。
    // 对于哈希映射中的每个键值对，键表示一个元素，值表示该元素出现的次数。

    // 用一个循环遍历数组 nums 并将数组中的每个元素加入哈希映射中。
    // 在这之后，遍历哈希映射中的所有键值对，返回值最大的键。
    // 同样也可以在遍历数组 nums 时候使用打擂台的方法，维护最大的值，
    // 这样省去了最后对哈希映射的遍历。
    int majorityElement(vector<int>& nums) {
        std::unordered_map<int, int> counts;
        int majority = 0;
        int max_cnt = 0;

        for (int num : nums) {
            counts[num]++;
            if (counts[num] > max_cnt) {
                majority = num;
                max_cnt = counts[num];
            }
        }

        return majority;
    }
};
// @lc code=end

