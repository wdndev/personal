/*
 * @lc app=leetcode.cn id=1929 lang=cpp
 *
 * [1929] 数组串联
 *
 * https://leetcode.cn/problems/concatenation-of-array/description/
 *
 * algorithms
 * Easy (85.02%)
 * Likes:    57
 * Dislikes: 0
 * Total Accepted:    43.2K
 * Total Submissions: 51.1K
 * Testcase Example:  '[1,2,1]'
 *
 * 给你一个长度为 n 的整数数组 nums 。请你构建一个长度为 2n 的答案数组 ans ，数组下标 从 0 开始计数 ，对于所有 0  的 i
 * ，满足下述所有要求：
 * 
 * 
 * ans[i] == nums[i]
 * ans[i + n] == nums[i]
 * 
 * 
 * 具体而言，ans 由两个 nums 数组 串联 形成。
 * 
 * 返回数组 ans 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [1,2,1]
 * 输出：[1,2,1,1,2,1]
 * 解释：数组 ans 按下述方式形成：
 * - ans = [nums[0],nums[1],nums[2],nums[0],nums[1],nums[2]]
 * - ans = [1,2,1,1,2,1]
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [1,3,2,1]
 * 输出：[1,3,2,1,1,3,2,1]
 * 解释：数组 ans 按下述方式形成：
 * - ans = [nums[0],nums[1],nums[2],nums[3],nums[0],nums[1],nums[2],nums[3]]
 * - ans = [1,3,2,1,1,3,2,1]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * n == nums.length
 * 1 
 * 1 
 * 
 * 
 */

// @lc code=start
class Solution1 {
public:
    vector<int> getConcatenation(vector<int>& nums) {
        std::vector<int> ans;
        for (auto& n : nums) {
            ans.push_back(n);
        }
        for (auto& n : nums) {
            ans.push_back(n);
        }

        return ans;
    }
};

class Solution {
public:
    vector<int> getConcatenation(vector<int>& nums) {
        int n = nums.size();
        for (int i = 0; i < n; ++i){
            nums.push_back(nums[i]);
        }
        return nums;
    }
};

// @lc code=end

