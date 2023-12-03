/*
 * @lc app=leetcode.cn id=4 lang=cpp
 *
 * [4] 寻找两个正序数组的中位数
 *
 * https://leetcode.cn/problems/median-of-two-sorted-arrays/description/
 *
 * algorithms
 * Hard (41.77%)
 * Likes:    6908
 * Dislikes: 0
 * Total Accepted:    1M
 * Total Submissions: 2.5M
 * Testcase Example:  '[1,3]\n[2]'
 *
 * 给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
 * 
 * 算法的时间复杂度应该为 O(log (m+n)) 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums1 = [1,3], nums2 = [2]
 * 输出：2.00000
 * 解释：合并数组 = [1,2,3] ，中位数 2
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums1 = [1,2], nums2 = [3,4]
 * 输出：2.50000
 * 解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
 * 
 * 
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * nums1.length == m
 * nums2.length == n
 * 0 <= m <= 1000
 * 0 <= n <= 1000
 * 1 <= m + n <= 2000
 * -10^6 <= nums1[i], nums2[i] <= 10^6
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.暴力，合并两个数组
    double findMedianSortedArrays1(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size();
        int n = nums2.size();
        std::vector<int> new_nums(m + n);
        if (m == 0) {
            return this->get_mid_num(nums2);
        }
        if (n == 0) {
            return this->get_mid_num(nums1);
        }

        int idx = 0;
        int num1_idx = 0;
        int num2_idx = 0;
        while (idx != m + n) {
            if (num1_idx == m) {
                while (num2_idx < n) {
                     new_nums[idx++] = nums2[num2_idx++];
                }
                break;
            }

            if (num2_idx == n) {
                while (num1_idx < m) {
                     new_nums[idx++] = nums1[num1_idx++];
                }
                break;
            }

            if (nums1[num1_idx] < nums2[num2_idx]) {
                new_nums[idx++] = nums1[num1_idx++];
            } else {
                new_nums[idx++] = nums2[num2_idx++];
            }
        }

        return this->get_mid_num(new_nums);
    }

    // 2.二分查找变形
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size();
        int n = nums2.size();
        int len = m + n;

        // 上一次循环的结果
        int left = -1;
        // 本次循环的结果
        int right = -1;

        int start_1 = 0;
        int start_2 = 0;

        for (int i = 0; i <= len / 2; i++) {
            left = right;
            if (start_1 < m && (start_2 >= n || nums1[start_1] < nums2[start_2])) {
                right = nums1[start_1++];
            } else {
                right = nums2[start_2++];
            }
        }

        if (len % 2 == 0) {
            return (left + right) / 2.0;
        } else {
            return right;
        }
    }

private:
    double get_mid_num(std::vector<int>& nums) {
        int num_size = nums.size();
        if (num_size % 2 == 0) {
            return (nums[num_size / 2 - 1] + nums[num_size / 2]) / 2.0;
        } else {
            return nums[num_size / 2];
        }
    }
};
// @lc code=end

