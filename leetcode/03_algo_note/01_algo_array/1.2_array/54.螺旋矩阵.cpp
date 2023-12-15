/*
 * @lc app=leetcode.cn id=54 lang=cpp
 *
 * [54] 螺旋矩阵
 *
 * https://leetcode.cn/problems/spiral-matrix/description/
 *
 * algorithms
 * Medium (49.92%)
 * Likes:    1559
 * Dislikes: 0
 * Total Accepted:    436.8K
 * Total Submissions: 874.5K
 * Testcase Example:  '[[1,2,3],[4,5,6],[7,8,9]]'
 *
 * 给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
 * 输出：[1,2,3,6,9,8,7,4,5]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
 * 输出：[1,2,3,4,8,12,11,10,9,5,6,7]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * m == matrix.length
 * n == matrix[i].length
 * 1 
 * -100 
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 按层模拟：可以将矩阵看成若干层，首先输出最外层的元素
    // 其次输出次外层的元素，直到输出最内层的元素
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        if (matrix.size() == 0 || matrix[0].size() == 0) {
            return {};
        }

        int row = matrix.size();
        int col = matrix[0].size();
        std::vector<int> order;

        int left = 0;
        int right = col - 1;
        int top = 0;
        int bottom = row - 1;

        while (left <= right && top <= bottom) {
            // 从左到右
            for (int i = left; i <= right; i++) {
                order.push_back(matrix[top][i]);
            }
            // 从上到下
            for (int j = top + 1; j <= bottom; j++) {
                order.push_back(matrix[j][right]);
            }
            // 现在idx在右下角，需要判断
            if (left < right && top < bottom) {
                // 从右到左
                for (int i = right - 1; i > left; i--) {
                    order.push_back(matrix[bottom][i]);
                }
                // 从上到下
                for (int j = bottom; j > top; j--) {
                    order.push_back(matrix[j][left]);
                }
            }

            left++;
            right--;
            top++;
            bottom--;
        }

        return order;
    }
};
// @lc code=end

