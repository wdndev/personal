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
    // 分治
    
    // # []
    // # [1]

    // # [2]
    // # [1, 2]

    // # [3]
    // # [1, 3]
    // # [2, 3]
    // # [1, 2, 3]
    vector<vector<int>> subsets(vector<int>& nums) {
        std::vector<std::vector<int>> ans;
        std::vector<int> list;
        if (nums.empty())
            return ans;

        this->dfs(ans, nums, list, 0);

        return ans;
    }
private:
    void dfs(std::vector<std::vector<int>>& ans, std::vector<int>& nums, vector<int> list, int idx) {
        // terminator
        if (idx == nums.size()) {
            ans.push_back(list);
            return;
        }

        // not pick the number at this idx
        this->dfs(ans, nums, list, idx + 1);

        // pick the number at this idx
        list.push_back(nums[idx]);
        this->dfs(ans, nums, list, idx + 1);
        list.pop_back();

    }
};
// @lc code=end

