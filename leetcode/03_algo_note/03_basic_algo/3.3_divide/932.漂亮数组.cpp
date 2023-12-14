/*
 * @lc app=leetcode.cn id=932 lang=cpp
 *
 * [932] 漂亮数组
 *
 * https://leetcode.cn/problems/beautiful-array/description/
 *
 * algorithms
 * Medium (66.16%)
 * Likes:    216
 * Dislikes: 0
 * Total Accepted:    14.5K
 * Total Submissions: 21.9K
 * Testcase Example:  '4'
 *
 * 如果长度为 n 的数组 nums 满足下述条件，则认为该数组是一个 漂亮数组 ：
 * 
 * 
 * nums 是由范围 [1, n] 的整数组成的一个排列。
 * 对于每个 0 <= i < j < n ，均不存在下标 k（i < k < j）使得 2 * nums[k] == nums[i] + nums[j]
 * 。
 * 
 * 
 * 给你整数 n ，返回长度为 n 的任一 漂亮数组 。本题保证对于给定的 n 至少存在一个有效答案。
 * 
 * 
 * 
 * 示例 1 ：
 * 
 * 
 * 输入：n = 4
 * 输出：[2,1,4,3]
 * 
 * 
 * 示例 2 ：
 * 
 * 
 * 输入：n = 5
 * 输出：[3,1,2,5,4]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= n <= 1000
 * 
 * 
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    vector<int> beautifulArray(int n) {
        if (n == 1) {
            return {1};
        }
        std::vector<int> ans(n, 0);
        int left_cnt = (n + 1) / 2;
        int right_cnt = n - left_cnt;

        std::vector<int> left_nums = this->beautifulArray(left_cnt);
        std::vector<int> right_nums = this->beautifulArray(right_cnt);
        // 奇数
        for (int i = 0; i < left_cnt; i++) {
            ans[i] = 2 * left_nums[i] - 1;
        }
        // 偶数
        for (int i = 0; i < right_cnt; i++) {
            ans[left_cnt + i] = 2 * right_nums[i];
        }

        return ans;
    }
};
// @lc code=end

