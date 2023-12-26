/*
 * @lc app=leetcode.cn id=704 lang=cpp
 *
 * [704] 二分查找
 *
 * https://leetcode.cn/problems/binary-search/description/
 *
 * algorithms
 * Easy (54.86%)
 * Likes:    1500
 * Dislikes: 0
 * Total Accepted:    1.1M
 * Total Submissions: 2M
 * Testcase Example:  '[-1,0,3,5,9,12]\n9'
 *
 * 给定一个 n 个元素有序的（升序）整型数组 nums 和一个目标值 target  ，写一个函数搜索 nums 中的
 * target，如果目标值存在返回下标，否则返回 -1。
 * 
 * 
 * 示例 1:
 * 
 * 输入: nums = [-1,0,3,5,9,12], target = 9
 * 输出: 4
 * 解释: 9 出现在 nums 中并且下标为 4
 * 
 * 
 * 示例 2:
 * 
 * 输入: nums = [-1,0,3,5,9,12], target = 2
 * 输出: -1
 * 解释: 2 不存在 nums 中因此返回 -1
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 你可以假设 nums 中的所有元素是不重复的。
 * n 将在 [1, 10000]之间。
 * nums 的每个元素都将在 [-9999, 9999]之间。
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0;
        int right = nums.size() - 1;
        // 在区间 [left, right] 内查找 target
        while (left <= right) {
            // 取区间中间节点
            int mid = left + (right - left) / 2;
            // 如果找到目标值，则直接返回中心位置
            if (nums[mid] == target) {
                return mid;
            // 如果 nums[mid] 小于目标值，则在 [mid + 1, right] 中继续搜索
            } else if (nums[mid] < target) {
                left = mid + 1;
            // 如果 nums[mid] 大于目标值，则在 [left, mid - 1] 中继续搜索
            } else {
                right = mid - 1;
            }
        }

        // 未搜索到元素，返回 -1
        return -1;
    }
};
// @lc code=end

