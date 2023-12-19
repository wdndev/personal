/*
 * @lc app=leetcode.cn id=46 lang=cpp
 *
 * [46] 全排列
 *
 * https://leetcode.cn/problems/permutations/description/
 *
 * algorithms
 * Medium (78.93%)
 * Likes:    2775
 * Dislikes: 0
 * Total Accepted:    974.4K
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
        std::vector<int> tmp_vec;
        this->dfs(0, tmp_vec, nums, ans);
        return ans;
    }

private:
    void dfs(int curr, std::vector<int>& tmp_vec, std::vector<int>& nums, std::vector<std::vector<int>>& ans) {
        // 终止条件
        if (curr == nums.size()) {
            ans.push_back(tmp_vec);
            return;
        }

        // 处理当前层
        for (int i = 0; i < nums.size(); i++) {
            // 从当前路径中没有出现的数字中选择
            if (std::find(tmp_vec.begin(), tmp_vec.end(), nums[i]) != tmp_vec.end()) {
                continue;
            }
            // 选择元素
            tmp_vec.push_back(nums[i]);
            // 继续递归下一层
            this->dfs(curr + 1, tmp_vec, nums, ans);
            // 撤销操作
            tmp_vec.pop_back();
        }
    }
};
// @lc code=end

