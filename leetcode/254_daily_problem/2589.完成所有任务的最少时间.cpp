/*
 * @lc app=leetcode.cn id=2589 lang=cpp
 *
 * [2589] 完成所有任务的最少时间
 *
 * https://leetcode.cn/problems/minimum-time-to-complete-all-tasks/description/
 *
 * algorithms
 * Hard (43.37%)
 * Likes:    72
 * Dislikes: 0
 * Total Accepted:    14K
 * Total Submissions: 24.5K
 * Testcase Example:  '[[2,3,1],[4,5,1],[1,5,2]]'
 *
 * 你有一台电脑，它可以 同时 运行无数个任务。给你一个二维整数数组 tasks ，其中 tasks[i] = [starti, endi,
 * durationi] 表示第 i 个任务需要在 闭区间 时间段 [starti, endi] 内运行 durationi
 * 个整数时间点（但不需要连续）。
 * 
 * 当电脑需要运行任务时，你可以打开电脑，如果空闲时，你可以将电脑关闭。
 * 
 * 请你返回完成所有任务的情况下，电脑最少需要运行多少秒。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 输入：tasks = [[2,3,1],[4,5,1],[1,5,2]]
 * 输出：2
 * 解释：
 * - 第一个任务在闭区间 [2, 2] 运行。
 * - 第二个任务在闭区间 [5, 5] 运行。
 * - 第三个任务在闭区间 [2, 2] 和 [5, 5] 运行。
 * 电脑总共运行 2 个整数时间点。
 * 
 * 
 * 示例 2：
 * 
 * 输入：tasks = [[1,3,2],[2,5,3],[5,6,2]]
 * 输出：4
 * 解释：
 * - 第一个任务在闭区间 [2, 3] 运行
 * - 第二个任务在闭区间 [2, 3] 和 [5, 5] 运行。
 * - 第三个任务在闭区间 [5, 6] 运行。
 * 电脑总共运行 4 个整数时间点。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= tasks.length <= 2000
 * tasks[i].length == 3
 * 1 <= starti, endi <= 2000
 * 1 <= durationi <= endi - starti + 1 
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 按照区间右端点从小到大排序
    // 排序后，对于区间[tasks[i]来说，它右侧任务区间要么有交集，要么包含它的一部分后缀
    // 遍历排序后的任务，先统计区间内的一运行的电脑运行时间点，如果个数小于duration，
    // 则需要新增时间点；
    // 尽量把新增的时间点安排在区间 [start, end]的后缀上，这样下一个区间就能统计更多运行时间点
    int findMinimumTime(vector<vector<int>>& tasks) {
        std::sort(tasks.begin(), tasks.end(), [](auto& a, auto& b) {
            return a[1] < b[1];
        });
        int ans = 0;
        int max_time = tasks[tasks.size() - 1][1];
        std::vector<bool> run(max_time + 1);
        for (auto& t : tasks) {
            int start = t[0];
            int end = t[1];
            int d = t[2];
            for (int i = start; i <= end; i++) {
                if (run[i]) {
                    // 去掉中间的运行点
                    d--;
                }
            }
            for (int i = end; d > 0; i--) {
                if (!run[i]) {
                    run[i] = true;
                    d--;
                    ans++;
                }
            }
        }
        return ans;
    }
};
// @lc code=end

