/*
 * @lc app=leetcode.cn id=1793 lang=cpp
 *
 * [1793] 好子数组的最大分数
 *
 * https://leetcode.cn/problems/maximum-score-of-a-good-subarray/description/
 *
 * algorithms
 * Hard (46.84%)
 * Likes:    122
 * Dislikes: 0
 * Total Accepted:    18.8K
 * Total Submissions: 34.7K
 * Testcase Example:  '[1,4,3,7,4,5]\n3'
 *
 * 给你一个整数数组 nums （下标从 0 开始）和一个整数 k 。
 * 
 * 一个子数组 (i, j) 的 分数 定义为 min(nums[i], nums[i+1], ..., nums[j]) * (j - i + 1)
 * 。一个 好 子数组的两个端点下标需要满足 i <= k <= j 。
 * 
 * 请你返回 好 子数组的最大可能 分数 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 输入：nums = [1,4,3,7,4,5], k = 3
 * 输出：15
 * 解释：最优子数组的左右端点下标是 (1, 5) ，分数为 min(4,3,7,4,5) * (5-1+1) = 3 * 5 = 15 。
 * 
 * 
 * 示例 2：
 * 
 * 输入：nums = [5,5,4,5,4,1,1,1], k = 0
 * 输出：20
 * 解释：最优子数组的左右端点下标是 (0, 4) ，分数为 min(5,5,4,5,4) * (4-0+1) = 4 * 5 = 20 。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= nums.length <= 10^5
 * 1 <= nums[i] <= 2 * 10^4
 * 0 <= k < nums.length
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 双指针
    // 例如 nums=[1,9,7,8,8,1], k=3。
    // 其中面积最大的矩形，左边界下标 L=1，右边界下标 R=4。
    // 尝试从 i=k, j=k 出发，通过不断移动指针来找到最大矩形。比较 nums[i−1] 和 nums[j+1] 的大小，
    // 谁大就移动谁（一样大移动哪个都可以）。
    // 在移动过程中，不断用 nums[i]和 nums[j]更新矩形高度的最小值 minH，
    // 同时用 minH⋅(j−i+1)更新答案的最大值。
    int maximumScore(vector<int>& nums, int k) {
        int n = nums.size();
        int ans = nums[k];
        int min_h = nums[k];
        int i = k, j = k;
        // 循环 n-1次
        for (int t = 0; t < n - 1; t++) {
            // 通过不断移动指针来找到最大矩形。
            // 比较 nums[i-1] 和 nums[j+1]的大小，那个大就移动那个
            if (j == n - 1 || i > 0 && nums[i - 1] > nums[j + 1]) {
                i--;
                min_h = std::min(min_h, nums[i]);
            } else {
                j++;
                min_h = std::min(min_h, nums[j]);
            }
            ans = std::max(ans, min_h * (j - i + 1));
        }

        return ans;
    }
};
// @lc code=end

