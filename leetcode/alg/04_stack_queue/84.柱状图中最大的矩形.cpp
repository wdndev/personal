// @before-stub-for-debug-begin
#include <vector>
#include <string>
#include "commoncppproblem84.h"

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=84 lang=cpp
 *
 * [84] 柱状图中最大的矩形
 *
 * https://leetcode.cn/problems/largest-rectangle-in-histogram/description/
 *
 * algorithms
 * Hard (45.22%)
 * Likes:    2589
 * Dislikes: 0
 * Total Accepted:    371.4K
 * Total Submissions: 821.2K
 * Testcase Example:  '[2,1,5,6,2,3]'
 *
 * 给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
 * 
 * 求在该柱状图中，能够勾勒出来的矩形的最大面积。
 * 
 * 
 * 
 * 示例 1:
 * 
 * 
 * 
 * 
 * 输入：heights = [2,1,5,6,2,3]
 * 输出：10
 * 解释：最大的矩形为图中红色区域，面积为 10
 * 
 * 
 * 示例 2：
 * 
 * 
 * 
 * 
 * 输入： heights = [2,4]
 * 输出： 4
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 
 * 0 
 * 
 * 
 */

// @lc code=start

// 1. 暴力求解 : O(n)， 枚举宽度
// for i -> 0, n-2 :
//     for j -> i+1, n-1:
//         (i, j) -> 最小高度, area 
//         update max_area

// 2.暴力求解2 ： 枚举高度
// for i -> 0, n-1:
//     找到 left bound, right_bound
//     area = height[i] * (right - left)
//     updata max_area

// 3.栈
        
class Solution {
public:
    // 1.固定高度，枚举宽度， 超时
    // 固定每个数组元素为每个矩形的高，然后遍历数组，寻找每个矩形高能构成的最大面积，
    // 当左右两边第一次出现比当前高小的元素值，即为当前高能构成的最大值，每次保存最大值
    int largestRectangleArea1(vector<int>& heights) {
        int max_area = 0;
        // 遍历高度
        for (int mid = 0; mid < heights.size(); mid++) {
            int h = heights[mid];
            int left = mid;
            int right = mid;
            // 左侧寻找最大宽度 
            while (left - 1 >= 0 && heights[left - 1] >= h) {
                left--;
            }
            // 右侧寻找最大宽度
            while (right + 1 < heights.size() && heights[right + 1] >= h) {
                right++;
            }

            max_area = std::max(max_area, h * (right - left + 1));
        }
        
        return max_area;
    }

    // 2.固定宽度，枚举高度, 超时
    // 固定左右两边的长度即固定宽的长度，然后遍历数组，寻找当前长度中高最短的元素，
    // 即当前宽能构成的最大矩形，每次保存最大值
    int largestRectangleArea2(vector<int>& heights) {
        int max_area = 0;
        int n = heights.size();
        if (n == 1) {
            return heights[0];
        }
        for (int left = 0; left < n; left++) {
            int min_height = heights[left];
            for (int right = left; right < n; right++) {
                min_height = std::min(min_height, heights[right]);
                max_area = std::max(max_area, min_height * (right - left + 1));
            }
            
        }

        return max_area;
    }

    // 3.单调栈
    // 在枚举宽的同时需要寻找高，在枚举高的时候又要寻找宽，时间消耗非常大
    // 那么可以利用递增栈优化暴力暴力求解的过程

    // 当元素大于栈顶元素时，入栈
    // 当元素小于栈顶元素时，维护栈的递增性，将小于当前元素的栈顶元素弹出，并计算面积
    int largestRectangleArea(vector<int>& heights) {
        int n = heights.size();
        if (n == 1) {
            return heights[0];
        }
        
        int max_area = 0;

        std::stack<int> stack;
        // 遍历数组
        for (int i = 0; i < n; i++) {
            while (!stack.empty() && heights[stack.top()] >= heights[i]) {
                // 出栈，并计算面积，维护递增性，需要对小于的元素全部出栈
                int length = heights[stack.top()];
                stack.pop();

                int weight = i;
                // 最后一个栈顶元素，出栈计算面积需要包含一下前面和后面，
                // 因为矩形可以延伸，这里需要好好想一想
                if (!stack.empty()) {
                    weight = i - stack.top() - 1;
                }

                max_area = std::max(max_area, length * weight);

            }
            // 入栈
            stack.push(i);
        }

        // 数组元素全部遍历完了，但是栈还有元素，进行清空栈
        while (!stack.empty()) {
            int length = heights[stack.top()];
            stack.pop();
            int weight = n;
            if (!stack.empty()) {
                weight = n - stack.top() - 1;
            }
            max_area = std::max(max_area, length * weight);
        }

        return max_area;
    }
};
// @lc code=end

