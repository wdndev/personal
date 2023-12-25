/*
 * @lc app=leetcode.cn id=75 lang=cpp
 *
 * [75] 颜色分类
 *
 * https://leetcode.cn/problems/sort-colors/description/
 *
 * algorithms
 * Medium (60.74%)
 * Likes:    1713
 * Dislikes: 0
 * Total Accepted:    588.4K
 * Total Submissions: 968K
 * Testcase Example:  '[2,0,2,1,1,0]'
 *
 * 给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
 * 
 * 我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。
 * 
 * 
 * 
 * 
 * 必须在不使用库内置的 sort 函数的情况下解决这个问题。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [2,0,2,1,1,0]
 * 输出：[0,0,1,1,2,2]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [2,0,1]
 * 输出：[0,1,2]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * n == nums.length
 * 1 <= n <= 300
 * nums[i] 为 0、1 或 2
 * 
 * 
 * 
 * 
 * 进阶：
 * 
 * 
 * 你能想出一个仅使用常数空间的一趟扫描算法吗？
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int left = 0;
        int right = nums.size() - 1;
        int idx = 0;

        while (idx <= right) {
            if (idx < left) {
                idx++;
            } else if (nums[idx] == 0) {
                swap(nums[idx], nums[left]);
                left++;
            } else if (nums[idx] == 2) {
                swap(nums[idx], nums[right]);
                right--;
            } else {
                idx++;
            }
        }
    }
};
// @lc code=end

