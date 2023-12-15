/*
 * @lc app=leetcode.cn id=498 lang=cpp
 *
 * [498] 对角线遍历
 *
 * https://leetcode.cn/problems/diagonal-traverse/description/
 *
 * algorithms
 * Medium (55.78%)
 * Likes:    470
 * Dislikes: 0
 * Total Accepted:    118.4K
 * Total Submissions: 212.2K
 * Testcase Example:  '[[1,2,3],[4,5,6],[7,8,9]]'
 *
 * 给你一个大小为 m x n 的矩阵 mat ，请以对角线遍历的顺序，用一个数组返回这个矩阵中的所有元素。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：mat = [[1,2,3],[4,5,6],[7,8,9]]
 * 输出：[1,2,4,7,5,3,6,8,9]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：mat = [[1,2],[3,4]]
 * 输出：[1,2,3,4]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * m == mat.length
 * n == mat[i].length
 * 1 <= m, n <= 10^4
 * 1 <= m * n <= 10^4
 * -10^5 <= mat[i][j] <= 10^5
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    vector<int> findDiagonalOrder(vector<vector<int>>& mat) {
        int row = mat.size();
        int col = mat[0].size();
        int count = row * col;
        std::vector<int> ans;

        // 起点
        int x = 0;
        int y = 0;

        for (int i = 0; i < row * col; i++) {
            ans.push_back(mat[x][y]);
            // 当「行号 + 列号」为偶数时，遍历方向为从左下到右上。
            // 可以记为右上方向 (−1,+1)，即行号减 1，列号加 1。
            if ((x + y) % 2 == 0) {
                // 最后一列，向下方移动
                if (y == col - 1) {
                    x++;
                // 第一行，向右方移动
                } else if (x == 0) {
                    y++;
                // 右上
                } else {
                    x--;
                    y++;
                }
            // 当「行号 + 列号」为奇数时，遍历方向为从右上到左下。
            // 可以记为左下方向 (+1,−1)，即行号加 1，列号减 1。
            } else {
                // 最后一行，向右方移动
                if (x == row - 1) {
                    y++;
                // 第一列，向下方移动
                } else if (y == 0) {
                    x++;
                // 左下方向
                } else {
                    x++;
                    y--;
                }
            }
        }

        return ans;
    }
};
// @lc code=end

