/*
 * @lc app=leetcode.cn id=1953 lang=cpp
 *
 * [1953] 你可以工作的最大周数
 *
 * https://leetcode.cn/problems/maximum-number-of-weeks-for-which-you-can-work/description/
 *
 * algorithms
 * Medium (38.15%)
 * Likes:    97
 * Dislikes: 0
 * Total Accepted:    21.3K
 * Total Submissions: 43.2K
 * Testcase Example:  '[1,2,3]'
 *
 * 给你 n 个项目，编号从 0 到 n - 1 。同时给你一个整数数组 milestones ，其中每个 milestones[i] 表示第 i
 * 个项目中的阶段任务数量。
 * 
 * 你可以按下面两个规则参与项目中的工作：
 * 
 * 
 * 每周，你将会完成 某一个 项目中的 恰好一个 阶段任务。你每周都 必须 工作。
 * 在 连续的 两周中，你 不能 参与并完成同一个项目中的两个阶段任务。
 * 
 * 
 * 一旦所有项目中的全部阶段任务都完成，或者仅剩余一个阶段任务都会导致你违反上面的规则，那么你将 停止工作
 * 。注意，由于这些条件的限制，你可能无法完成所有阶段任务。
 * 
 * 返回在不违反上面规则的情况下你 最多 能工作多少周。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：milestones = [1,2,3]
 * 输出：6
 * 解释：一种可能的情形是：
 * ​​​​- 第 1 周，你参与并完成项目 0 中的一个阶段任务。
 * - 第 2 周，你参与并完成项目 2 中的一个阶段任务。
 * - 第 3 周，你参与并完成项目 1 中的一个阶段任务。
 * - 第 4 周，你参与并完成项目 2 中的一个阶段任务。
 * - 第 5 周，你参与并完成项目 1 中的一个阶段任务。
 * - 第 6 周，你参与并完成项目 2 中的一个阶段任务。
 * 总周数是 6 。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：milestones = [5,2,1]
 * 输出：7
 * 解释：一种可能的情形是：
 * - 第 1 周，你参与并完成项目 0 中的一个阶段任务。
 * - 第 2 周，你参与并完成项目 1 中的一个阶段任务。
 * - 第 3 周，你参与并完成项目 0 中的一个阶段任务。
 * - 第 4 周，你参与并完成项目 1 中的一个阶段任务。
 * - 第 5 周，你参与并完成项目 0 中的一个阶段任务。
 * - 第 6 周，你参与并完成项目 2 中的一个阶段任务。
 * - 第 7 周，你参与并完成项目 0 中的一个阶段任务。
 * 总周数是 7 。
 * 注意，你不能在第 8 周参与完成项目 0 中的最后一个阶段任务，因为这会违反规则。
 * 因此，项目 0 中会有一个阶段任务维持未完成状态。
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * n == milestones.length
 * 1 <= n <= 10^5
 * 1 <= milestones[i] <= 10^9
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 贪心
    // 考虑时长最长的工作。假设需要longest周完成该工作，其余工作需要rest周完成，
    // 那么可完成工作的充要条件是 longest <= rest + 1
    // 比较两个大小：
    // - longest <= rest + 1，所有工作都可以完成，返回总耗时：longest + rest
    // - longest > rest + 1，此时，完不成最耗时的工作，耗时最长的工作可以完成 rest+1周，
    //   因此最大工作周数 2 * rest + 1
    long long numberOfWeeks(vector<int>& milestones) {
        int longest = 0;
        long long rest = 0;
        for (auto& m : milestones) {
            longest = std::max(longest, m);
            rest += m;
        }

        // 其余工作需要完成的周数
        rest -= longest;
        if (longest > rest + 1) {
            // 此时，无法完成所耗时最长的工作
            return rest * 2 + 1;
        } else {
            // 可以完成
            return longest + rest;
        }
    }
};
// @lc code=end

