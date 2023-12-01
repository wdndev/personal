/*
 * @lc app=leetcode.cn id=78 lang=cpp
 *
 * [78] 子集
 *
 * https://leetcode.cn/problems/subsets/description/
 *
 * algorithms
 * Medium (81.15%)
 * Likes:    2171
 * Dislikes: 0
 * Total Accepted:    690.1K
 * Total Submissions: 850.4K
 * Testcase Example:  '[1,2,3]'
 *
 * 给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
 * 
 * 解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [1,2,3]
 * 输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
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
 * nums 中的所有元素 互不相同
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        std::vector<std::vector<int>> ans;
        std::vector<int> tmp_vec;
        if (nums.size() == 0) {
            return ans;
        }

        this->dfs(0, nums, tmp_vec, ans);

        return ans;
    }

    void dfs(int curr, std::vector<int>& nums, std::vector<int> tmp_vec, 
        std::vector<std::vector<int>>& ans) {
        if (curr == nums.size()) {
            ans.push_back(tmp_vec);
            return;
        }

        // 不选则当前结点值
        this->dfs(curr + 1, nums, tmp_vec, ans);

        // 选择当前结点值
        tmp_vec.push_back(nums[curr]);
        this->dfs(curr + 1, nums, tmp_vec, ans);
        tmp_vec.pop_back();
    }
};
// @lc code=end

