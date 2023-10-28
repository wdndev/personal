/*
 * @lc app=leetcode.cn id=213 lang=cpp
 *
 * [213] 打家劫舍 II
 *
 * https://leetcode.cn/problems/house-robber-ii/description/
 *
 * algorithms
 * Medium (44.71%)
 * Likes:    1512
 * Dislikes: 0
 * Total Accepted:    387.8K
 * Total Submissions: 867.4K
 * Testcase Example:  '[2,3,2]'
 *
 * 你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 围成一圈
 * ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警 。
 * 
 * 给定一个代表每个房屋存放金额的非负整数数组，计算你 在不触动警报装置的情况下 ，今晚能够偷窃到的最高金额。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [2,3,2]
 * 输出：3
 * 解释：你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [1,2,3,1]
 * 输出：4
 * 解释：你可以先偷窃 1 号房屋（金额 = 1），然后偷窃 3 号房屋（金额 = 3）。
 * 偷窃到的最高金额 = 1 + 3 = 4 。
 * 
 * 示例 3：
 * 
 * 
 * 输入：nums = [1,2,3]
 * 输出：3
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= nums.length <= 100
 * 0 <= nums[i] <= 1000
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 如何才能保证第一间房屋和最后一间房屋不同时偷窃呢？
    // 如果偷窃了第一间房屋，则不能偷窃最后一间房屋，因此偷窃房屋的范围是第一间房屋到最后第二间房屋；
    // 如果偷窃了最后一间房屋，则不能偷窃第一间房屋，因此偷窃房屋的范围是第二间房屋到最后一间房屋。

    // 假设数组 nums 的长度为 n。

    // - 如果不偷窃最后一间房屋，则偷窃房屋的下标范围是 `[0,n−2]`；
    // - 如果不偷窃第一间房屋，则偷窃房屋的下标范围是 `[1,n−1]`。

    // 在确定偷窃房屋的下标范围之后，即可用第 198 题的方法解决。
    // 对于两段下标范围分别计算可以偷窃到的最高总金额，
    // 其中的最大值即为在 n 间房屋中可以偷窃到的最高总金额。
    int rob(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) {
            return 0;
        } else if (n == 1) {
            return nums[0];
        } else if (n == 2) {
            return std::max(nums[0], nums[1]);
        }

        return std::max(this->rob_range(nums, 0, n - 2), this->rob_range(nums, 1, n - 1));
    }
private:
    int rob_range(std::vector<int>& nums, int start, int end) {
        std::vector<int> dp(nums.size(), 0);

        dp[start] = nums[start];
        dp[start + 1] = std::max(nums[start], nums[start + 1]);
        int res = dp[start + 1];
        for (int i = start + 2; i <= end; i++) {
            dp[i] = std::max(dp[i - 1] + 0, dp[i - 2] + nums[i]);

            res = std::max(res, dp[i]);
        }

        return res;
    }
};
// @lc code=end

