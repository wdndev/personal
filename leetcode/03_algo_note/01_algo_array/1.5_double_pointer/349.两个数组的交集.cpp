/*
 * @lc app=leetcode.cn id=349 lang=cpp
 *
 * [349] 两个数组的交集
 *
 * https://leetcode.cn/problems/intersection-of-two-arrays/description/
 *
 * algorithms
 * Easy (74.23%)
 * Likes:    866
 * Dislikes: 0
 * Total Accepted:    512.3K
 * Total Submissions: 689.8K
 * Testcase Example:  '[1,2,2,1]\n[2,2]'
 *
 * 给定两个数组 nums1 和 nums2 ，返回 它们的交集 。输出结果中的每个元素一定是 唯一 的。我们可以 不考虑输出结果的顺序 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums1 = [1,2,2,1], nums2 = [2,2]
 * 输出：[2]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
 * 输出：[9,4]
 * 解释：[4,9] 也是可通过的
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= nums1.length, nums2.length <= 1000
 * 0 <= nums1[i], nums2[i] <= 1000
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        // 排序
        std::sort(nums1.begin(), nums1.end());
        std::sort(nums2.begin(), nums2.end());

        std::vector<int> ans;

        int idx_1 = 0;
        int idx_2 = 0;
        while (idx_1 < nums1.size() && idx_2 < nums2.size()) {
            if (nums1[idx_1] == nums2[idx_2]) {
                // 保证加入元素的唯一性
                if (!ans.size() || nums1[idx_1] != ans.back()) {
                    ans.push_back(nums1[idx_1]);
                }
                idx_1++;
                idx_2++;
            } else if (nums1[idx_1] < nums2[idx_2]) {
                idx_1++;
            } else {
                idx_2++;
            }
        }

        return ans;
    }
};
// @lc code=end

