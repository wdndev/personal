/*
 * @lc app=leetcode.cn id=826 lang=cpp
 *
 * [826] 安排工作以达到最大收益
 *
 * https://leetcode.cn/problems/most-profit-assigning-work/description/
 *
 * algorithms
 * Medium (43.50%)
 * Likes:    153
 * Dislikes: 0
 * Total Accepted:    34.1K
 * Total Submissions: 68.1K
 * Testcase Example:  '[2,4,6,8,10]\n[10,20,30,40,50]\n[4,5,6,7]'
 *
 * 你有 n 个工作和 m 个工人。给定三个数组： difficulty, profit 和 worker ，其中:
 * 
 * 
 * difficulty[i] 表示第 i 个工作的难度，profit[i] 表示第 i 个工作的收益。
 * worker[i] 是第 i 个工人的能力，即该工人只能完成难度小于等于 worker[i] 的工作。
 * 
 * 
 * 每个工人 最多 只能安排 一个 工作，但是一个工作可以 完成多次 。
 * 
 * 
 * 举个例子，如果 3 个工人都尝试完成一份报酬为 $1 的同样工作，那么总收益为 $3 。如果一个工人不能完成任何工作，他的收益为 $0 。
 * 
 * 
 * 返回 在把工人分配到工作岗位后，我们所能获得的最大利润 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入: difficulty = [2,4,6,8,10], profit = [10,20,30,40,50], worker = [4,5,6,7]
 * 输出: 100 
 * 解释: 工人被分配的工作难度是 [4,4,6,6] ，分别获得 [20,20,30,30] 的收益。
 * 
 * 示例 2:
 * 
 * 
 * 输入: difficulty = [85,47,57], profit = [24,66,99], worker = [40,25,25]
 * 输出: 0
 * 
 * 
 * 
 * 提示:
 * 
 * 
 * n == difficulty.length
 * n == profit.length
 * m == worker.length
 * 1 <= n, m <= 10^4
 * 1 <= difficulty[i], profit[i], worker[i] <= 10^5
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 把 difficulty 和 profit 绑在一起，按照 difficulty 从小到大排序，
    // 我们可以在遍历worker的同时，用双指针遍历并维护 difficulty[j] <= worker[i] 的最大 profit[j]
    // 即第i个工人所能获得的最大利润
    int maxProfitAssignment(vector<int>& difficulty, vector<int>& profit, vector<int>& worker) {
        int n = difficulty.size();
        std::vector<std::pair<int, int>> jobs(n);
        for (int i = 0; i < n; i++) {
            jobs[i] = {difficulty[i], profit[i]};
        }

        std::sort(jobs.begin(), jobs.end());
        std::sort(worker.begin(), worker.end());

        int ans = 0;
        int job_idx = 0;
        int max_profit = 0;
        // 注意此粗，max_profit 不在for循环中，表示当前的 worker 可以去做之前利润高的工作
        for (auto& w : worker) {
            while (job_idx < n && jobs[job_idx].first <= w)
            {
                max_profit = std::max(max_profit, jobs[job_idx].second);
                job_idx++;
            }
            ans += max_profit;
        }

        return ans;
    }
};
// @lc code=end

