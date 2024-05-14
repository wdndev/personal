/*
 * @lc app=leetcode.cn id=2244 lang=cpp
 *
 * [2244] 完成所有任务需要的最少轮数
 *
 * https://leetcode.cn/problems/minimum-rounds-to-complete-all-tasks/description/
 *
 * algorithms
 * Medium (57.43%)
 * Likes:    47
 * Dislikes: 0
 * Total Accepted:    30.7K
 * Total Submissions: 47.2K
 * Testcase Example:  '[2,2,3,3,2,4,4,4,4,4]'
 *
 * 给你一个下标从 0 开始的整数数组 tasks ，其中 tasks[i] 表示任务的难度级别。在每一轮中，你可以完成 2 个或者 3 个 相同难度级别
 * 的任务。
 * 
 * 返回完成所有任务需要的 最少 轮数，如果无法完成所有任务，返回 -1 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 输入：tasks = [2,2,3,3,2,4,4,4,4,4]
 * 输出：4
 * 解释：要想完成所有任务，一个可能的计划是：
 * - 第一轮，完成难度级别为 2 的 3 个任务。 
 * - 第二轮，完成难度级别为 3 的 2 个任务。 
 * - 第三轮，完成难度级别为 4 的 3 个任务。 
 * - 第四轮，完成难度级别为 4 的 2 个任务。 
 * 可以证明，无法在少于 4 轮的情况下完成所有任务，所以答案为 4 。
 * 
 * 
 * 示例 2：
 * 
 * 输入：tasks = [2,3,3]
 * 输出：-1
 * 解释：难度级别为 2 的任务只有 1 个，但每一轮执行中，只能选择完成 2 个或者 3 个相同难度级别的任务。因此，无法完成所有任务，答案为 -1
 * 。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= tasks.length <= 10^5
 * 1 <= tasks[i] <= 10^9
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 每轮完成都是相同难度级别的任务，如果假设难度为1的任务有c个，问题变成：
    // - 每轮可以把 c 减少 2，或者减少3. 把 c 减少到 0 最少要多少轮？
    // 贪心：尽量多使用减少3，可以让轮数尽量少
    // 分类讨论：
    // - 如果 c = 1，无法完成，返回 -1
    // - 如果 c = 3k (k > 1)，只用减少 3 就能完成，轮数为 c/3
    // - 如果 c = 3k + 1 (k >= 1), 即 c = 3k + 4,可以把c减少到4，然后使用两次 -2，轮数为 (c-4)/3 + 2 = (c+2)/3= upper(c / 3)
    // - 如果 c = 3k + 2, 可以先减少到2，再用一次-2，轮数为 (c - 2)/3 + 1 = (c + 1) / 3 = upper(c / 3)
    // 综上，对于 c >= 2个相同难度的任务，最少需要操作 upper(c/3)=lower((c+2) / 3)
    // 用哈希表统计不同难度的任务个数，按照上式计算，累加轮数
    int minimumRounds(vector<int>& tasks) {
        std::unordered_map<int, int> cnt;
        for (auto& t : tasks) {
            cnt[t]++;
        }
        int ans = 0;
        for (auto& it : cnt) {
            int c = it.second;
            if (c == 1) {
                return -1;
            }
            ans += (c + 2) / 3;
        }

        return ans;
    }
};
// @lc code=end

