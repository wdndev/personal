/*
 * @lc app=leetcode.cn id=1673 lang=cpp
 *
 * [1673] 找出最具竞争力的子序列
 *
 * https://leetcode.cn/problems/find-the-most-competitive-subsequence/description/
 *
 * algorithms
 * Medium (40.79%)
 * Likes:    174
 * Dislikes: 0
 * Total Accepted:    27.2K
 * Total Submissions: 55.6K
 * Testcase Example:  '[3,5,2,6]\n2'
 *
 * 给你一个整数数组 nums 和一个正整数 k ，返回长度为 k 且最具 竞争力 的 nums 子序列。
 * 
 * 数组的子序列是从数组中删除一些元素（可能不删除元素）得到的序列。
 * 
 * 在子序列 a 和子序列 b 第一个不相同的位置上，如果 a 中的数字小于 b 中对应的数字，那么我们称子序列 a 比子序列 b（相同长度下）更具 竞争力
 * 。 例如，[1,3,4] 比 [1,3,5] 更具竞争力，在第一个不相同的位置，也就是最后一个位置上， 4 小于 5 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [3,5,2,6], k = 2
 * 输出：[2,6]
 * 解释：在所有可能的子序列集合 {[3,5], [3,2], [3,6], [5,2], [5,6], [2,6]} 中，[2,6] 最具竞争力。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [2,4,3,3,5,4,9,6], k = 4
 * 输出：[2,3,3,4]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 
 * 0 
 * 1 
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 题意：返回 nums 的长度恰好为k的字典序最小子序列
    // 1.创建一个空栈
    // 2.从左遍历 nums
    // 3.设 x=nums[i]。如果栈不为空，且x小于栈顶，且栈的大小加上剩余元素个数 (n-i) 大于k
    //   则可以弹出栈顶。不断循环，指导不满足这三个条件之一
    // 4.如果栈的大小小于k，把x入栈
    // 5.遍历结束，栈（从底到顶）就是答案
    vector<int> mostCompetitive(vector<int>& nums, int k) {
        std::vector<int> stk;

        for (int i = 0; i < nums.size(); i++) {
            int x = nums[i];
            while (!stk.empty() && x < stk.back() && stk.size() + nums.size() - i > k) {
                stk.pop_back();
            }
            if (stk.size() < k) {
                stk.push_back(x);
            }
        }

        return stk;
    }
};
// @lc code=end

