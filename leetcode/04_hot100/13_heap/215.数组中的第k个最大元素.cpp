// @before-stub-for-debug-begin
#include <vector>
#include <string>
#include "commoncppproblem215.h"

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=215 lang=cpp
 *
 * [215] 数组中的第K个最大元素
 *
 * https://leetcode.cn/problems/kth-largest-element-in-an-array/description/
 *
 * algorithms
 * Medium (62.53%)
 * Likes:    2361
 * Dislikes: 0
 * Total Accepted:    960.1K
 * Total Submissions: 1.5M
 * Testcase Example:  '[3,2,1,5,6,4]\n2'
 *
 * 给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。
 * 
 * 请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。
 * 
 * 你必须设计并实现时间复杂度为 O(n) 的算法解决此问题。
 * 
 * 
 * 
 * 示例 1:
 * 
 * 
 * 输入: [3,2,1,5,6,4], k = 2
 * 输出: 5
 * 
 * 
 * 示例 2:
 * 
 * 
 * 输入: [3,2,3,1,2,4,5,5,6], k = 4
 * 输出: 4
 * 
 * 
 * 
 * 提示： 
 * 
 * 
 * 1 <= k <= nums.length <= 10^5
 * -10^4 <= nums[i] <= 10^4
 * 
 * 
 */

// @lc code=start
// 1.快速排序
class Solution1 {
public:
    // 1.快速排序
    int findKthLargest(vector<int>& nums, int k) {
        return this->quick_select(nums, 0, nums.size() - 1, nums.size() - k);
    }
    int quick_select(std::vector<int>& nums, int left, int right, int k) {
        if (left == right) {
            return nums[k];
        }

        int pivot = nums[left];
        int i = left - 1;
        int j = right + 1;
        while (i < j)
        {
            do {
                i++;
            } while (nums[i] < pivot);

            do {
                j--;
            } while (nums[j] > pivot);

            if (i < j) {
                swap(nums[i], nums[j]);
            }
        }
        if (k <= j) {
            return this->quick_select(nums, left, j, k);
        } else {
            return this->quick_select(nums, j + 1, right, k);
        }
    }
};

// 2.堆排序
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        int heap_size = nums.size();
        this->build_max_heap(nums, heap_size);
        for (int i = nums.size() - 1; i >= nums.size() - k + 1; i--) {
            swap(nums[0], nums[i]);
            heap_size--;
            this->max_heapify(nums, 0, heap_size);
        }

        return nums[0];
    }
    
    // 从上往下堆化
    void max_heapify(std::vector<int>& nums, int i, int heap_size) {
        // 父节点，左子树，右子树
        int largest_idx = i;
        int left_idx = 2 * i + 1;
        int right_idx = 2 * i + 2;
        
        // 寻找左右子树两个子节点中最大的一个
        if (left_idx < heap_size && nums[left_idx] > nums[largest_idx]) {
            largest_idx = left_idx;
        }
        if (right_idx < heap_size && nums[right_idx] > nums[largest_idx]) {
            largest_idx = right_idx;
        }
        // 如果能找到子节点比自己大
        // 交换元素，继续堆化
        if (largest_idx != i) {
            swap(nums[i], nums[largest_idx]);
            this->max_heapify(nums, largest_idx, heap_size);
        }
    }

    void build_max_heap(std::vector<int>& nums, int heap_size) {
        for (int i = heap_size / 2; i >= 0; i--) {
            this->max_heapify(nums, i, heap_size);
        }
    }
};
// @lc code=end

