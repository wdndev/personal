/*
 * @lc app=leetcode.cn id=2951 lang=cpp
 *
 * [2951] 找出峰值
 *
 * https://leetcode.cn/problems/find-the-peaks/description/
 *
 * algorithms
 * Easy (79.16%)
 * Likes:    19
 * Dislikes: 0
 * Total Accepted:    26.2K
 * Total Submissions: 31.3K
 * Testcase Example:  '[2,4,4]'
 *
 * 给你一个下标从 0 开始的数组 mountain 。你的任务是找出数组 mountain 中的所有 峰值。
 * 
 * 以数组形式返回给定数组中 峰值 的下标，顺序不限 。
 * 
 * 注意：
 * 
 * 
 * 峰值 是指一个严格大于其相邻元素的元素。
 * 数组的第一个和最后一个元素 不 是峰值。
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：mountain = [2,4,4]
 * 输出：[]
 * 解释：mountain[0] 和 mountain[2] 不可能是峰值，因为它们是数组的第一个和最后一个元素。
 * mountain[1] 也不可能是峰值，因为它不严格大于 mountain[2] 。
 * 因此，答案为 [] 。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：mountain = [1,4,3,8,5]
 * 输出：[1,3]
 * 解释：mountain[0] 和 mountain[4] 不可能是峰值，因为它们是数组的第一个和最后一个元素。
 * mountain[2] 也不可能是峰值，因为它不严格大于 mountain[3] 和 mountain[1] 。
 * 但是 mountain[1] 和 mountain[3] 严格大于它们的相邻元素。
 * 因此，答案是 [1,3] 。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 3 <= mountain.length <= 100
 * 1 <= mountain[i] <= 100
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    vector<int> findPeaks(vector<int>& mountain) {
        std::vector<int> ans;
        for (int i = 1; i < mountain.size() - 1; i++) {
            if (mountain[i] > mountain[i - 1] && mountain[i] > mountain[i + 1]) {
                ans.push_back(i);
            }
        }

        return ans;
    }
};
// @lc code=end

