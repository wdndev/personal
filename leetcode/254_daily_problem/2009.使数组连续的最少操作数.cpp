/*
 * @lc app=leetcode.cn id=2009 lang=cpp
 *
 * [2009] 使数组连续的最少操作数
 *
 * https://leetcode.cn/problems/minimum-number-of-operations-to-make-array-continuous/description/
 *
 * algorithms
 * Hard (44.17%)
 * Likes:    86
 * Dislikes: 0
 * Total Accepted:    15.1K
 * Total Submissions: 29.4K
 * Testcase Example:  '[4,2,5,3]'
 *
 * 给你一个整数数组 nums 。每一次操作中，你可以将 nums 中 任意 一个元素替换成 任意 整数。
 * 
 * 如果 nums 满足以下条件，那么它是 连续的 ：
 * 
 * 
 * nums 中所有元素都是 互不相同 的。
 * nums 中 最大 元素与 最小 元素的差等于 nums.length - 1 。
 * 
 * 
 * 比方说，nums = [4, 2, 5, 3] 是 连续的 ，但是 nums = [1, 2, 3, 5, 6] 不是连续的 。
 * 
 * 请你返回使 nums 连续 的 最少 操作次数。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 输入：nums = [4,2,5,3]
 * 输出：0
 * 解释：nums 已经是连续的了。
 * 
 * 
 * 示例 2：
 * 
 * 输入：nums = [1,2,3,5,6]
 * 输出：1
 * 解释：一个可能的解是将最后一个元素变为 4 。
 * 结果数组为 [1,2,3,5,4] ，是连续数组。
 * 
 * 
 * 示例 3：
 * 
 * 输入：nums = [1,10,100,1000]
 * 输出：3
 * 解释：一个可能的解是：
 * - 将第二个元素变为 2 。
 * - 将第三个元素变为 3 。
 * - 将第四个元素变为 4 。
 * 结果数组为 [1,2,3,4] ，是连续数组。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= nums.length <= 10^5
 * 1 <= nums[i] <= 10^9
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 正难则反 + 滑动窗口
    // 正难则反：考虑最多保留多少个元素不变
    // 设 x 是修改后的正连续数字的最大值，则修改后的连续数字的范围为闭区间 [x-n+1, x],其中n是nums的长度。
    // 在修改前，对于已经在 [x-n+1, x]中的数，无需修改。那么x取多少，可以让无需修改的数最多？
    // 元素位置不影响答案，可以先将 nums 排序并去掉重复元素
    // 此时，把排序后的画在数轴上，相当于有一个长度为n的滑动窗口，我们需要计算窗口内最多可以包含多少个数轴上的点
    // 只需要枚举 排序后数组 作为窗口的右端点
    // 为了计算出窗口内有多少个点，先需要知道包含最左边的点在哪，设这个点位置为 a[left]，则必须大于等于窗口的左边界：a[left] >= a[i]-n+1
    // 此时，窗口内有 i-left+1 个点，取其最大值，就得到了最多保留不变的元素个数。
    // 最后用n减去保留不变的元素个数，就得到答案
    int minOperations(vector<int>& nums) {
        int n = nums.size();
        // 先去重
        std::set<int> s(nums.begin(), nums.end());
        nums.assign(s.begin(), s.end());
        // 排序
        std::sort(nums.begin(), nums.end());
        int m = nums.size();

        int ans = 0;
        int left = 0;
        for (int i = 0; i < m; i++) {
            // nums[left] 不在窗口内
            while (nums[left] < nums[i] - n + 1) {
                left++;
            }
            ans = std::max(ans, i - left + 1);
        }
        
        return n - ans;
    }
};
// @lc code=end

