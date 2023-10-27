/*
 * @lc app=leetcode.cn id=153 lang=cpp
 *
 * [153] 寻找旋转排序数组中的最小值
 *
 * https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/description/
 *
 * algorithms
 * Medium (57.15%)
 * Likes:    1037
 * Dislikes: 0
 * Total Accepted:    404.2K
 * Total Submissions: 707.2K
 * Testcase Example:  '[3,4,5,1,2]'
 *
 * 已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,2,4,5,6,7]
 * 在变化后可能得到：
 * 
 * 若旋转 4 次，则可以得到 [4,5,6,7,0,1,2]
 * 若旋转 7 次，则可以得到 [0,1,2,4,5,6,7]
 * 
 * 
 * 注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2],
 * ..., a[n-2]] 。
 * 
 * 给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。
 * 
 * 你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [3,4,5,1,2]
 * 输出：1
 * 解释：原数组为 [1,2,3,4,5] ，旋转 3 次得到输入数组。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [4,5,6,7,0,1,2]
 * 输出：0
 * 解释：原数组为 [0,1,2,4,5,6,7] ，旋转 4 次得到输入数组。
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：nums = [11,13,15,17]
 * 输出：11
 * 解释：原数组为 [11,13,15,17] ，旋转 4 次得到输入数组。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * n == nums.length
 * 1 <= n <= 5000
 * -5000 <= nums[i] <= 5000
 * nums 中的所有整数 互不相同
 * nums 原来是一个升序排序的数组，并进行了 1 至 n 次旋转
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.暴力
    int findMin1(vector<int>& nums) {
        int min_num = INT_MAX;
        for (auto& n : nums) {
            min_num = std::min(min_num, n);
        }

        return min_num;
    }

    // 2.二分查找
    // 在二分查找的每一步中，左边界为 left，右边界为 right，区间的中点为 mid，最小值就在该区间内。
    // 将中轴元素 nums[mid]与右边界元素 nums[right]进行比较，可能会有以下的三种情况：

    // 第一种情况是 nums[mid]<nums[right]。这说明 nums[mid] 是最小值右侧的元素，因此可以忽略二分查找区间的右半部分。
    // 第二种情况是 nums[mid]>nums[right]。这说明 nums[mid]是最小值左侧的元素，因此可以忽略二分查找区间的左半部分。
    // 由于数组不包含重复元素，并且只要当前的区间长度不为 1，mid 就不会与 right 重合；
    // 而如果当前的区间长度为 1，这说明已经可以结束二分查找了。因此不会存在 nums[mid]=nums[right]的情况。

    // 当二分查找结束时，我们就得到了最小值所在的位置。
    int findMin(vector<int>& nums) {
        int left = 0;
        int right = nums.size() - 1;
        int idx = -1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            int mid_val = nums[mid];
            if (mid_val < nums[right]) {
                right = mid;
            } else if (mid_val > nums[right]) {
                left = mid + 1;
            }
        }

        return nums[left];
    }


};
// @lc code=end

