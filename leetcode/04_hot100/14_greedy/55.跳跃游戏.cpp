/*
 * @lc app=leetcode.cn id=55 lang=cpp
 *
 * [55] 跳跃游戏
 *
 * https://leetcode.cn/problems/jump-game/description/
 *
 * algorithms
 * Medium (43.35%)
 * Likes:    2604
 * Dislikes: 0
 * Total Accepted:    826.7K
 * Total Submissions: 1.9M
 * Testcase Example:  '[2,3,1,1,4]'
 *
 * 给你一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。
 * 
 * 判断你是否能够到达最后一个下标，如果可以，返回 true ；否则，返回 false 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [2,3,1,1,4]
 * 输出：true
 * 解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [3,2,1,0,4]
 * 输出：false
 * 解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= nums.length <= 10^4
 * 0 <= nums[i] <= 10^5
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.暴力求解O(n^2)， 超时
    // 使用额外的一个数组，初始化为false，依据不同的下标，进行跳跃，
    // 同时修改数组位置为true，最后观察数组最后一个位置是不是true
    bool canJump1(vector<int>& nums) {
        if (nums.size() == 0) {
            return false;
        }
        std::vector<int> flag_arr(nums.size(), 0);
        if (nums[0] == 0 && nums.size() > 1) {
            return false;
        }
        flag_arr[0] = 1;
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] == 0) {
                continue;
            }
            for (int j = i + 1; j <= i + nums[i]; j++) {
                if (j < nums.size()) {
                    flag_arr[j] = 1;
                }
            }
        }

        return flag_arr[nums.size() - 1] == 1;
    }

    // 2.贪心算法：从后往前贪心
    bool canJump(vector<int>& nums) {
        if (nums.size() == 0) {
            return false;
        }

        int end_reachable = nums.size() - 1;

        for (int i = nums.size() - 1; i >= 0; i--) {
            if (nums[i] + i >= end_reachable) {
                end_reachable = i;
            }
        }

        return end_reachable == 0;
    } 
};
// @lc code=end

