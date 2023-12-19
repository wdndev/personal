/*
 * @lc app=leetcode.cn id=90 lang=cpp
 *
 * [90] 子集 II
 *
 * https://leetcode.cn/problems/subsets-ii/description/
 *
 * algorithms
 * Medium (63.46%)
 * Likes:    1178
 * Dislikes: 0
 * Total Accepted:    338.3K
 * Total Submissions: 533K
 * Testcase Example:  '[1,2,2]'
 *
 * 给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的子集（幂集）。
 * 
 * 解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。
 * 
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [1,2,2]
 * 输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [0]
 * 输出：[[],[0]]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 
 * -10 
 * 
 * 
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        std::vector<std::vector<int>> ans;
        std::vector<int> tmp_vec;
        if (nums.size() == 0) {
            return ans;
        }
        // 排序，用于后续剪枝
        std::sort(nums.begin(), nums.end());

        this->dfs(0, nums, tmp_vec, ans);

        return ans;
    }

    void dfs(int curr, std::vector<int>& nums, std::vector<int> tmp_vec, 
        std::vector<std::vector<int>>& ans) {
        // 加入结果中
        ans.push_back(tmp_vec);

        for (int i = curr; i < nums.size(); i++) {
            // 去重
            if (i > curr && nums[i] == nums[i - 1]) {
                continue;
            }
            // 可以通过向当前子集数组中添加可选元素来表示选择该元素。
            // 也可以在当前递归结束之后，
            // 将之前添加的元素从当前子集数组中移除（也就是回溯）来表示不选择该元素。
            tmp_vec.push_back(nums[i]);
            this->dfs(i + 1, nums, tmp_vec, ans);
            tmp_vec.pop_back();
        }
    }
};
// @lc code=end

