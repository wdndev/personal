/*
 * @lc app=leetcode.cn id=40 lang=cpp
 *
 * [40] 组合总和 II
 *
 * https://leetcode.cn/problems/combination-sum-ii/description/
 *
 * algorithms
 * Medium (59.51%)
 * Likes:    1494
 * Dislikes: 0
 * Total Accepted:    483.5K
 * Total Submissions: 812.6K
 * Testcase Example:  '[10,1,2,7,6,1,5]\n8'
 *
 * 给定一个候选人编号的集合 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
 * 
 * candidates 中的每个数字在每个组合中只能使用 一次 。
 * 
 * 注意：解集不能包含重复的组合。 
 * 
 * 
 * 
 * 示例 1:
 * 
 * 
 * 输入: candidates = [10,1,2,7,6,1,5], target = 8,
 * 输出:
 * [
 * [1,1,6],
 * [1,2,5],
 * [1,7],
 * [2,6]
 * ]
 * 
 * 示例 2:
 * 
 * 
 * 输入: candidates = [2,5,2,1,2], target = 5,
 * 输出:
 * [
 * [1,2,2],
 * [5]
 * ]
 * 
 * 
 * 
 * 提示:
 * 
 * 
 * 1 <= candidates.length <= 100
 * 1 <= candidates[i] <= 50
 * 1 <= target <= 30
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        std::vector<std::vector<int>> ans;
        std::vector<int> tmp_vec;
        if (candidates.size() == 0) {
            return ans;
        }

        // 排序，方便后续剪枝
        std::sort(candidates.begin(), candidates.end());

        this->dfs(0, candidates, target, tmp_vec, ans);
        return ans;
    }

    void dfs(int curr, std::vector<int>& candidates, int target, 
        std::vector<int> tmp_vec, std::vector<std::vector<int>>& ans) {
        // 终止条件，等于0，因为小于0的被剪枝完了
        if (target == 0) {
            ans.push_back(tmp_vec);
            return;
        } else if (target < 0 || curr >= candidates.size()) {
            return;
        }

        for (int i = curr; i < candidates.size(); i++) {
            // 因为数组提前排序了，这儿可以直接剪枝
            if (target - candidates[i] < 0) {
                break;
            }
            // 去重
            if (i > curr && candidates[i] == candidates[i - 1]) {
                continue;
            }

            tmp_vec.push_back(candidates[i]);
            this->dfs(i + 1, candidates, target - candidates[i], tmp_vec, ans);
            tmp_vec.pop_back();
        }
    }
};
// @lc code=end

