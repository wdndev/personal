/*
 * @lc app=leetcode.cn id=11 lang=cpp
 *
 * [11] 盛最多水的容器
 *
 * https://leetcode.cn/problems/container-with-most-water/description/
 *
 * algorithms
 * Medium (60.01%)
 * Likes:    4763
 * Dislikes: 0
 * Total Accepted:    1.1M
 * Total Submissions: 1.9M
 * Testcase Example:  '[1,8,6,2,5,4,8,3,7]'
 *
 * 给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。
 * 
 * 找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
 * 
 * 返回容器可以储存的最大水量。
 * 
 * 说明：你不能倾斜容器。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 
 * 
 * 输入：[1,8,6,2,5,4,8,3,7]
 * 输出：49 
 * 解释：图中垂直线代表输入数组 [1,8,6,2,5,4,8,3,7]。在此情况下，容器能够容纳水（表示为蓝色部分）的最大值为 49。
 * 
 * 示例 2：
 * 
 * 
 * 输入：height = [1,1]
 * 输出：1
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * n == height.length
 * 2 <= n <= 10^5
 * 0 <= height[i] <= 10^4
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.枚举： left bar x, right bar y, (x-y)*height_diff
    // 超出时间限制
    // O(n^2)
    int maxArea1(vector<int>& height) {
        int max_water = 0;
        for (int i = 0; i < height.size(); i++) {
            for (int j = i + 1; j < height.size(); j++) {
                int tmp = std::min(height[i], height[j]) * (j - i);
                max_water = std::max(max_water, tmp);
            }
        }

        return max_water;
    }

    // 2.双指针
    // left和right两个指针，那么bar的高度小，那个往里面移动
    // O(n)
    int maxArea(vector<int>& height) {
        int left = 0;
        int right = height.size() - 1;
        int max_water = 0;

        while (left < right) {
            int tmp = std::min(height[left], height[right]) * (right - left);
            max_water = std::max(max_water, tmp);

            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }

        return max_water;
    }
};
// @lc code=end

