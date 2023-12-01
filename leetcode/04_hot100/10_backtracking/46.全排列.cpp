/*
 * @lc app=leetcode.cn id=46 lang=cpp
 *
 * [46] 全排列
 *
 * https://leetcode.cn/problems/permutations/description/
 *
 * algorithms
 * Medium (78.95%)
 * Likes:    2726
 * Dislikes: 0
 * Total Accepted:    946.1K
 * Total Submissions: 1.2M
 * Testcase Example:  '[1,2,3]'
 *
 * 给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [1,2,3]
 * 输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [0,1]
 * 输出：[[0,1],[1,0]]
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：nums = [1]
 * 输出：[[1]]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= nums.length <= 6
 * -10 <= nums[i] <= 10
 * nums 中的所有整数 互不相同
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        std::vector<std::vector<int>> ans;
        this->dfs(0, nums, ans);
        return ans;
    }

private:
    void dfs(int curr, std::vector<int>& nums, std::vector<std::vector<int>>& ans) {
        // 终止条件
        if (curr == nums.size()) {
            ans.push_back(nums);
            return;
        }

        // 处理当前层
        for (int i = curr; i < nums.size(); i++) {
            // 动态维护数组
            swap(nums[i], nums[curr]);
            // 继续递归下一层
            this->dfs(curr + 1, nums, ans);
            // 撤销操作
            swap(nums[i], nums[curr]);
        }
    }
};
// @lc code=end

