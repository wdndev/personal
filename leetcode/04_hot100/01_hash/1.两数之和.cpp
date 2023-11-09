/*
 * @lc app=leetcode.cn id=1 lang=cpp
 *
 * [1] 两数之和
 *
 * https://leetcode.cn/problems/two-sum/description/
 *
 * algorithms
 * Easy (53.01%)
 * Likes:    17828
 * Dislikes: 0
 * Total Accepted:    4.9M
 * Total Submissions: 9.2M
 * Testcase Example:  '[2,7,11,15]\n9'
 *
 * 给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
 * 
 * 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
 * 
 * 你可以按任意顺序返回答案。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [2,7,11,15], target = 9
 * 输出：[0,1]
 * 解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [3,2,4], target = 6
 * 输出：[1,2]
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：nums = [3,3], target = 6
 * 输出：[0,1]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 2 <= nums.length <= 10^4
 * -10^9 <= nums[i] <= 10^9
 * -10^9 <= target <= 10^9
 * 只会存在一个有效答案
 * 
 * 
 * 
 * 
 * 进阶：你可以想出一个时间复杂度小于 O(n^2) 的算法吗？
 * 
 */

// @lc code=start
class Solution {
public:
    // 暴力搜索
    vector<int> twoSum1(vector<int>& nums, int target) {
        int size = nums.size();

        for (int i = 0; i < size; i++) {
            for (int j = i + 1; j < size; j++) {
                if (nums[i] + nums[j] == target) {
                    return {i, j};
                }
            }
        }

        return {};
    }

    // 哈希表
    // 创建一个哈希表，对于每一个 x，我们首先查询哈希表中是否存在 target - x，
    // 然后将 x 插入到哈希表中，即可保证不会让 x 和自己匹配。
    vector<int> twoSum(vector<int>& nums, int target) {
        std::unordered_map<int, int> hash_table;

        for (int i = 0; i < nums.size(); i++) {
            auto it = hash_table.find(target - nums[i]);
            if (it != hash_table.end()) {
                return {i, it->second};
            }
            hash_table[nums[i]] = i;
        }

        return {};
    }
    
};
// @lc code=end

