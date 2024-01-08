/*
 * @lc app=leetcode.cn id=739 lang=cpp
 *
 * [739] 每日温度
 *
 * https://leetcode.cn/problems/daily-temperatures/description/
 *
 * algorithms
 * Medium (68.78%)
 * Likes:    1681
 * Dislikes: 0
 * Total Accepted:    497.3K
 * Total Submissions: 723K
 * Testcase Example:  '[73,74,75,71,69,72,76,73]'
 *
 * 给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指对于第 i
 * 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 0 来代替。
 * 
 * 
 * 
 * 示例 1:
 * 
 * 
 * 输入: temperatures = [73,74,75,71,69,72,76,73]
 * 输出: [1,1,4,2,1,1,0,0]
 * 
 * 
 * 示例 2:
 * 
 * 
 * 输入: temperatures = [30,40,50,60]
 * 输出: [1,1,1,0]
 * 
 * 
 * 示例 3:
 * 
 * 
 * 输入: temperatures = [30,60,90]
 * 输出: [1,1,0]
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= temperatures.length <= 10^5
 * 30 <= temperatures[i] <= 100
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.暴力解法
    vector<int> dailyTemperatures1(vector<int>& temperatures) {
        int n = temperatures.size();
        std::vector<int> ans(n);
        // 记录每个温度第一次出现的下标
        std::vector<int> next(101, INT_MAX);

        // 反向遍历
        for (int i = n - 1; i >= 0; i--) {
            int warmer_idx = INT_MAX;
            // 对于每个元素 temperatures[i]，在数组 next 中找到从 
            // temperatures[i] + 1 到 100 中每个温度第一次出现的下标
            // warmer_index 为下一次温度比当天高的下标
            for (int t = temperatures[i] + 1; t <= 100; t++) {
                warmer_idx = std::min(warmer_idx, next[t]);
            }

            if (warmer_idx != INT_MAX) {
                ans[i] = warmer_idx - i;
            }

            next[temperatures[i]] = i;
        }

        return ans;
    }

    // 2.单调栈
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        int n = temperatures.size();
        std::vector<int> ans(n);
        std::stack<int> stk;

        for (int i = 0; i < n; i++) {
            while (!stk.empty() && temperatures[i] > temperatures[stk.top()]) {
                int previous_idx = stk.top();
                ans[previous_idx] = i - previous_idx;
                stk.pop();
            }
            stk.push(i);
        }

        return ans;
    }
};
// @lc code=end

