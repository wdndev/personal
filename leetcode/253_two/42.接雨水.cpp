// @before-stub-for-debug-begin
#include <vector>
#include <string>
#include "commoncppproblem42.h"

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=42 lang=cpp
 *
 * [42] 接雨水
 *
 * https://leetcode.cn/problems/trapping-rain-water/description/
 *
 * algorithms
 * Hard (63.48%)
 * Likes:    5145
 * Dislikes: 0
 * Total Accepted:    951.4K
 * Total Submissions: 1.5M
 * Testcase Example:  '[0,1,0,2,1,0,1,3,2,1,2,1]'
 *
 * 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 
 * 
 * 输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
 * 输出：6
 * 解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：height = [4,2,0,3,2,5]
 * 输出：9
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * n == height.length
 * 1 <= n <= 2 * 10^4
 * 0 <= height[i] <= 10^5
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 当前列可以承载的雨滴数量 = min(当前列左右两侧的最大值) - 当前列的大小
    // 朴素的做法是对于数组 height 中的每个元素，分别向左和向右扫描并记录左边和右边的最大高度，
    // 然后计算每个下标位置能接的雨水量。
    // 创建两个长度为 n 的数组 leftMax 和 rightMax。
    // 对于 0≤i<n，leftMax[i]表示下标 i 及其左边的位置中，
    // height的最大高度，rightMax[i] 表示下标 i 及其右边的位置中，height的最大高度。
    int trap(vector<int>& height) {
        int n = height.size();
        if (n == 0) {
            return 0;
        }
        // 向右扫描，记录第i点左侧的最大高度
        std::vector<int> left_max(n);
        left_max[0] = height[0];
        for (int i = 1; i < n; i++) {
            left_max[i] = std::max(left_max[i - 1], height[i]);
        }

        // 相左扫描，记录第i点左侧的最大高度
        std::vector<int> right_max(n);
        right_max[n - 1] = height[n - 1];
        for (int i = n - 2; i >= 0; i--) {
            right_max[i] = std::max(right_max[i + 1], height[i]);
        }

        // 下标i处能接的雨水总量：min(left_max[i], right_max[i]) - height[i]
        int ans = 0;
        for (int i = 0; i < n; i++) {
            ans += std::min(left_max[i], right_max[i]) - height[i];
        }

        return ans;
    }
};
// @lc code=end

