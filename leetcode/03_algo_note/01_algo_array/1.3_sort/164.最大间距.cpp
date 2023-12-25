/*
 * @lc app=leetcode.cn id=164 lang=cpp
 *
 * [164] 最大间距
 *
 * https://leetcode.cn/problems/maximum-gap/description/
 *
 * algorithms
 * Medium (60.04%)
 * Likes:    607
 * Dislikes: 0
 * Total Accepted:    90.9K
 * Total Submissions: 151.3K
 * Testcase Example:  '[3,6,9,1]'
 *
 * 给定一个无序的数组 nums，返回 数组在排序之后，相邻元素之间最大的差值 。如果数组元素个数小于 2，则返回 0 。
 * 
 * 您必须编写一个在「线性时间」内运行并使用「线性额外空间」的算法。
 * 
 * 
 * 
 * 示例 1:
 * 
 * 
 * 输入: nums = [3,6,9,1]
 * 输出: 3
 * 解释: 排序后的数组是 [1,3,6,9], 其中相邻元素 (3,6) 和 (6,9) 之间都存在最大差值 3。
 * 
 * 示例 2:
 * 
 * 
 * 输入: nums = [10]
 * 输出: 0
 * 解释: 数组元素个数小于 2，因此返回 0。
 * 
 * 
 * 
 * 提示:
 * 
 * 
 * 1 <= nums.length <= 10^5
 * 0 <= nums[i] <= 10^9
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    int maximumGap(vector<int>& nums) {
        if (nums.size() < 2) {
            return 0;
        }
        std::vector<int> sort_nums = this->radix_sort(nums);
        int max_gap = 0;
        for (int i = 1; i < sort_nums.size(); i++) {
            max_gap = std::max(max_gap, sort_nums[i] - sort_nums[i-1]);
        }

        // for (auto& n : sort_nums) {
        //     std::cout << n << " " ;
        // }
        // std::cout << std::endl;

        return max_gap;
    }

    std::vector<int> radix_sort(std::vector<int>& nums) {
        int max_val = *max_element(nums.begin(), nums.end());
        int max_size = std::to_string(max_val).size();

        std::vector<int> sort_nums;
        for (int i = 0; i < max_size; i++) {
            std::vector<std::vector<int>> buckets(10, std::vector<int>());
            for (auto& n : nums) {
                buckets[int(n / pow(10, i)) % 10].push_back(n);
            }
            nums.clear();
            for (auto& bucket : buckets) {
                for (auto& n : bucket) {
                    nums.push_back(n);
                }
            }
        }

        return nums;
    }
};
// @lc code=end

