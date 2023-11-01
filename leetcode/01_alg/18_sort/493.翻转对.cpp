// @before-stub-for-debug-begin
#include <vector>
#include <string>
#include "commoncppproblem493.h"

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=493 lang=cpp
 *
 * [493] 翻转对
 *
 * https://leetcode.cn/problems/reverse-pairs/description/
 *
 * algorithms
 * Hard (36.64%)
 * Likes:    429
 * Dislikes: 0
 * Total Accepted:    42.8K
 * Total Submissions: 116.7K
 * Testcase Example:  '[1,3,2,3,1]'
 *
 * 给定一个数组 nums ，如果 i < j 且 nums[i] > 2*nums[j] 我们就将 (i, j) 称作一个重要翻转对。
 * 
 * 你需要返回给定数组中的重要翻转对的数量。
 * 
 * 示例 1:
 * 
 * 
 * 输入: [1,3,2,3,1]
 * 输出: 2
 * 
 * 
 * 示例 2:
 * 
 * 
 * 输入: [2,4,3,5,1]
 * 输出: 3
 * 
 * 
 * 注意:
 * 
 * 
 * 给定数组的长度不会超过50000。
 * 输入数组中的所有数字都在32位整数的表示范围内。
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    int reversePairs(vector<int>& nums) {
        if (nums.size() == 0) {
            return 0;
        }
        return this->merge_sort(nums, 0, nums.size() - 1);
    }
private:
    int merge_sort(std::vector<int>& nums, int left, int right) {
        if (left >= right) {
            return 0;
        }

        int mid = left + (right - left) / 2;
        int count = this->merge_sort(nums, left, mid) + this->merge_sort(nums, mid + 1, right);

        // 临时数组，排序后使用
        std::vector<int> sorted(right - left + 1);
        int i = left;
        int t = left;
        int c = 0;
        for (int j = mid + 1; j <= right; j++, c++) {
            while (i <= mid && (long long)nums[i] <= 2 * (long long)nums[j]) {
                i++;
            }
            while (t <= mid && nums[t] < nums[j]) {
                sorted[c++] = nums[t++];
            }
            sorted[c] = nums[j];
            count += (mid - i + 1);
        }

        while (t <= mid) {
            sorted[c++] = nums[t++];
        }

        for (int k = 0; k < sorted.size(); k++) {
            nums[left + k] = sorted[k];
        }

        return count;
    }
};
// @lc code=end

