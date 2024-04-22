/*
 * @lc app=leetcode.cn id=377 lang=cpp
 *
 * [377] 组合总和 Ⅳ
 *
 * https://leetcode.cn/problems/combination-sum-iv/description/
 *
 * algorithms
 * Medium (52.61%)
 * Likes:    992
 * Dislikes: 0
 * Total Accepted:    200.8K
 * Total Submissions: 379.8K
 * Testcase Example:  '[1,2,3]\n4'
 *
 * 给你一个由 不同 整数组成的数组 nums ，和一个目标整数 target 。请你从 nums 中找出并返回总和为 target 的元素组合的个数。
 * 
 * 题目数据保证答案符合 32 位整数范围。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [1,2,3], target = 4
 * 输出：7
 * 解释：
 * 所有可能的组合为：
 * (1, 1, 1, 1)
 * (1, 1, 2)
 * (1, 2, 1)
 * (1, 3)
 * (2, 1, 1)
 * (2, 2)
 * (3, 1)
 * 请注意，顺序不同的序列被视作不同的组合。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [9], target = 3
 * 输出：0
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 
 * 1 
 * nums 中的所有元素 互不相同
 * 1 
 * 
 * 
 * 
 * 
 * 进阶：如果给定的数组中含有负数会发生什么？问题会产生何种变化？如果允许负数出现，需要向题目中添加哪些限制条件？
 * 
 */

// @lc code=start
class Solution {
public:
    // 记忆化搜索
    int combinationSum4(vector<int>& nums, int target) {
        // -1 表示没有计算过
        std::vector<int> memo(target + 1, -1);

        return this->dfs(target, nums, memo);
    }

    int dfs(int i, std::vector<int>& nums, std::vector<int>& memo) {
        if (i == 0) {
            return 1;
        }
        int res = memo[i];
        if (res != -1) {
            return res;
        }
        int tmp_res = 0;
        for (int x : nums) {
            if (x <= i) {
                tmp_res += this->dfs(i - x, nums, memo);
            }
        }
        memo[i] = tmp_res;
        return tmp_res;
    }
};
// @lc code=end

