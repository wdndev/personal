/*
 * @lc app=leetcode.cn id=416 lang=cpp
 *
 * [416] 分割等和子集
 *
 * https://leetcode.cn/problems/partition-equal-subset-sum/description/
 *
 * algorithms
 * Medium (52.28%)
 * Likes:    1945
 * Dislikes: 0
 * Total Accepted:    478.6K
 * Total Submissions: 915.3K
 * Testcase Example:  '[1,5,11,5]'
 *
 * 给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [1,5,11,5]
 * 输出：true
 * 解释：数组可以分割成 [1, 5, 5] 和 [11] 。
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [1,2,3,5]
 * 输出：false
 * 解释：数组不能分割成两个元素和相等的子集。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 
 * 1 
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int sum_num = accumulate(nums.begin(), nums.end(), 0);
        if (sum_num%2 == 1) {
            return false;
        }
        int target = sum_num / 2;
        return this->zero_ont_pack_method2(nums, nums, target) == target;
    }

    // 思路2：动态规划 + 滚动数组优化
    int zero_ont_pack_method2(std::vector<int>& weight, std::vector<int>& value, int W) {
        int size = weight.size();
        std::vector<int> dp(W+1, 0);

        // 枚举前i种物品
        for (int i = 1; i <= size; i++) {
            // 逆序枚举背包装载重量（避免状态值错误）
            for (int w = W; w > weight[i - 1] - 1; w--) {
                // dp[w] 取「前 i - 1 件物品装入载重为 w 的背包中的最大价值」
                // 与「前 i - 1 件物品装入载重为 w - weight[i - 1] 的背包中，
                // 再装入第 i - 1 物品所得的最大价值」两者中的最大值
                dp[w] = std::max(dp[w], dp[w-weight[i-1]] + value[i-1]);
            }
        }
        return dp[W];
    }
};
// @lc code=end

