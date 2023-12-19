/*
 * @lc app=leetcode.cn id=47 lang=cpp
 *
 * [47] 全排列 II
 *
 * https://leetcode.cn/problems/permutations-ii/description/
 *
 * algorithms
 * Medium (65.59%)
 * Likes:    1510
 * Dislikes: 0
 * Total Accepted:    510.3K
 * Total Submissions: 777.9K
 * Testcase Example:  '[1,1,2]'
 *
 * 给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [1,1,2]
 * 输出：
 * [[1,1,2],
 * ⁠[1,2,1],
 * ⁠[2,1,1]]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [1,2,3]
 * 输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= nums.length <= 8
 * -10 <= nums[i] <= 10
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        m_ans.clear();
        m_tmp_vec.clear();
        std::vector<bool> visited(nums.size(), false);
        sort(nums.begin(), nums.end());
        // 排序
        this->dfs(0, nums, visited);
        return m_ans;
    }
private:
    void dfs(int curr, std::vector<int>& nums, std::vector<bool>& visited) {
        // 终止条件
        if (curr == nums.size()) {
            m_ans.push_back(m_tmp_vec);
            return;
        }

        // 处理当前层
        for (int i = 0; i < nums.size(); i++) {
            // 1.当前结点访问过
            // 2.跳过重复元素
            if (visited[i] || (i > 0 && nums[i] == nums[i - 1]) && !visited[i-1]) {
                continue;
            }
            visited[i] = true;
            // 选择元素
            m_tmp_vec.push_back(nums[i]);
            // 继续递归下一层
            this->dfs(curr + 1, nums, visited);
            // 撤销操作
            m_tmp_vec.pop_back();
            visited[i] = false;
        }
    }

    std::vector<std::vector<int>> m_ans;
    std::vector<int> m_tmp_vec;
};
// @lc code=end

