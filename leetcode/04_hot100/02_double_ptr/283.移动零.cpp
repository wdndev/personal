// @before-stub-for-debug-begin
#include <vector>
#include <string>
#include "commoncppproblem283.h"

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=283 lang=cpp
 *
 * [283] 移动零
 *
 * https://leetcode.cn/problems/move-zeroes/description/
 *
 * algorithms
 * Easy (63.61%)
 * Likes:    2206
 * Dislikes: 0
 * Total Accepted:    1.2M
 * Total Submissions: 1.8M
 * Testcase Example:  '[0,1,0,3,12]'
 *
 * 给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
 * 
 * 请注意 ，必须在不复制数组的情况下原地对数组进行操作。
 * 
 * 
 * 
 * 示例 1:
 * 
 * 
 * 输入: nums = [0,1,0,3,12]
 * 输出: [1,3,12,0,0]
 * 
 * 
 * 示例 2:
 * 
 * 
 * 输入: nums = [0]
 * 输出: [0]
 * 
 * 
 * 
 * 提示:
 * 
 * 
 * 
 * 1 <= nums.length <= 10^4
 * -2^31 <= nums[i] <= 2^31 - 1
 * 
 * 
 * 
 * 
 * 进阶：你能尽量减少完成的操作次数吗？
 * 
 */

// @lc code=start
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        // j 记录每次循环中的非零元素下标
        int j = 0;
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] != 0) {
                nums[j] = nums[i];

                // 处理最后一个数
                if (i != j) {
                    nums[i] = 0;
                }
                j++;
            }
        }
    }
};
// @lc code=end

