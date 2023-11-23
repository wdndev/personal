/*
 * @lc app=leetcode.cn id=56 lang=cpp
 *
 * [56] 合并区间
 *
 * https://leetcode.cn/problems/merge-intervals/description/
 *
 * algorithms
 * Medium (49.63%)
 * Likes:    2149
 * Dislikes: 0
 * Total Accepted:    733.4K
 * Total Submissions: 1.5M
 * Testcase Example:  '[[1,3],[2,6],[8,10],[15,18]]'
 *
 * 以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi]
 * 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
 * 输出：[[1,6],[8,10],[15,18]]
 * 解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：intervals = [[1,4],[4,5]]
 * 输出：[[1,5]]
 * 解释：区间 [1,4] 和 [4,5] 可被视为重叠区间。
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= intervals.length <= 10^4
 * intervals[i].length == 2
 * 0 <= starti <= endi <= 10^4
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 左端点排序处理
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        if (intervals.size() == 0) {
            return {};
        }
        // 左端点排序
        std::sort(intervals.begin(), intervals.end());

        // 合并数组结果
        std::vector<std::vector<int>> merged;

        // 遍历数组
        for (int i = 0; i < intervals.size(); i++) {
            int left = intervals[i][0];
            int right = intervals[i][1];

            // 如果merged为空，或是merged最后一个的右端点小于当前的左端点，加入数组
            // 佛则，进行比较，选择最大的
            if (!merged.size() || merged.back()[1] < left) {
                merged.push_back({left, right});
            } else {
                merged.back()[1] = std::max(merged.back()[1], right);
            }
        }

        return merged;
    }
};
// @lc code=end

