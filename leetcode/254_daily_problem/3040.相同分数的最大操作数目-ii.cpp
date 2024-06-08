/*
 * @lc app=leetcode.cn id=3040 lang=cpp
 *
 * [3040] 相同分数的最大操作数目 II
 *
 * https://leetcode.cn/problems/maximum-number-of-operations-with-the-same-score-ii/description/
 *
 * algorithms
 * Medium (34.53%)
 * Likes:    38
 * Dislikes: 0
 * Total Accepted:    12K
 * Total Submissions: 27.2K
 * Testcase Example:  '[3,2,1,2,3,4]'
 *
 * 给你一个整数数组 nums ，如果 nums 至少 包含 2 个元素，你可以执行以下操作中的 任意 一个：
 * 
 * 
 * 选择 nums 中最前面两个元素并且删除它们。
 * 选择 nums 中最后两个元素并且删除它们。
 * 选择 nums 中第一个和最后一个元素并且删除它们。
 * 
 * 
 * 一次操作的 分数 是被删除元素的和。
 * 
 * 在确保 所有操作分数相同 的前提下，请你求出 最多 能进行多少次操作。
 * 
 * 请你返回按照上述要求 最多 可以进行的操作次数。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [3,2,1,2,3,4]
 * 输出：3
 * 解释：我们执行以下操作：
 * - 删除前两个元素，分数为 3 + 2 = 5 ，nums = [1,2,3,4] 。
 * - 删除第一个元素和最后一个元素，分数为 1 + 4 = 5 ，nums = [2,3] 。
 * - 删除第一个元素和最后一个元素，分数为 2 + 3 = 5 ，nums = [] 。
 * 由于 nums 为空，我们无法继续进行任何操作。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [3,2,6,1,4]
 * 输出：2
 * 解释：我们执行以下操作：
 * - 删除前两个元素，分数为 3 + 2 = 5 ，nums = [6,1,4] 。
 * - 删除最后两个元素，分数为 1 + 4 = 5 ，nums = [6] 。
 * 至多进行 2 次操作。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 2 <= nums.length <= 2000
 * 1 <= nums[i] <= 1000
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // dfs
    // dfs[i,j] 表示下标在闭区间 [i, j]内的连续数组，最多可以执行多少次
    // 枚举这三种操作，分别从 dfs(i+2, j) + 1, dfs(i, j-2)+1, dfs(i+1, j-1)+1转移过来，取最大值，即为dfs[i,j]
    // 终止条件：i >= j
    int maxOperations(vector<int>& nums) {
        int n = nums.size();
        std::vector<std::vector<int>> memo(n, std::vector<int>(n, -1));
        int ans1 = this->dfs(2, n - 1, nums[0] + nums[1], nums, memo);

        for (auto& v : memo) {
            std::fill(v.begin(), v.end(), -1);
        }
        int ans2 = this->dfs(0, n - 3, nums[n-1] + nums[n-2], nums, memo);

        for (auto& v : memo) {
            std::fill(v.begin(), v.end(), -1);
        }
        int ans3 = this->dfs(1, n - 2, nums[0] + nums[n-1], nums, memo);


        return std::max({ans1, ans2, ans3}) + 1;
    }

    int dfs(int i, int j, int target, std::vector<int>& nums, std::vector<std::vector<int>>& memo) {
        if (i >= j) {
            return 0;
        }

        if (memo[i][j] != -1) {
            return memo[i][j];
        }
        memo[i][j] = 0;
        if (nums[i] + nums[i + 1] == target) {
            memo[i][j] = std::max(memo[i][j], this->dfs(i + 2, j, target, nums, memo) + 1);
        }

        if (nums[j - 1] + nums[j] == target) {
            memo[i][j] = std::max(memo[i][j], this->dfs(i, j - 2, target, nums, memo) + 1);
        }

        if (nums[i] + nums[j] == target) {
            memo[i][j] = std::max(memo[i][j], this->dfs(i + 1, j - 1, target, nums, memo) + 1);
        }

        return memo[i][j];
    }
};
// @lc code=end

