/*
 * @lc app=leetcode.cn id=220 lang=cpp
 *
 * [220] 存在重复元素 III
 *
 * https://leetcode.cn/problems/contains-duplicate-iii/description/
 *
 * algorithms
 * Hard (30.35%)
 * Likes:    721
 * Dislikes: 0
 * Total Accepted:    98.8K
 * Total Submissions: 325.2K
 * Testcase Example:  '[1,2,3,1]\n3\n0'
 *
 * 给你一个整数数组 nums 和两个整数 indexDiff 和 valueDiff 。
 * 
 * 找出满足下述条件的下标对 (i, j)：
 * 
 * 
 * i != j,
 * abs(i - j) <= indexDiff
 * abs(nums[i] - nums[j]) <= valueDiff
 * 
 * 
 * 如果存在，返回 true ；否则，返回 false 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [1,2,3,1], indexDiff = 3, valueDiff = 0
 * 输出：true
 * 解释：可以找出 (i, j) = (0, 3) 。
 * 满足下述 3 个条件：
 * i != j --> 0 != 3
 * abs(i - j) <= indexDiff --> abs(0 - 3) <= 3
 * abs(nums[i] - nums[j]) <= valueDiff --> abs(1 - 1) <= 0
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [1,5,9,1,5,9], indexDiff = 2, valueDiff = 3
 * 输出：false
 * 解释：尝试所有可能的下标对 (i, j) ，均无法满足这 3 个条件，因此返回 false 。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 2 <= nums.length <= 10^5
 * -10^9 <= nums[i] <= 10^9
 * 1 <= indexDiff <= nums.length
 * 0 <= valueDiff <= 10^9
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.桶排序
    int get_id(int x, int w) {
        return x < 0 ? (x + 1) / w - 1 : x / w;
    }
    bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {
        std::unordered_map<int, int> map;
        for (int i = 0; i < nums.size(); i++) {
            // 将 nums[i] 划分到 t + 1个桶中
            int id = get_id(nums[i], (t + 1));
            // 桶中已经有元素
            if (map.count(id)) {
                return true;
            }

            // 把nums[i]放入桶中
            map[id] = nums[i];

            //判断左侧桶是否满足条件
            if (map.count(id - 1) && abs(nums[i] - map[id - 1]) <= t) {
                return true;
            }
            // 判断右侧桶是否满足条件
            if (map.count(id + 1) && abs(nums[i] - map[id + 1]) <= t) {
                return true;
            }

            // 将 i - indexDiff 之前的桶清除，因为之前的桶已经不满足条件了
            if (i >= k) {
                map.erase(get_id(nums[i - k] , (t + 1)));
            }
        }
        return false;
    }
};
// @lc code=end

