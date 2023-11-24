/*
 * @lc app=leetcode.cn id=240 lang=cpp
 *
 * [240] 搜索二维矩阵 II
 *
 * https://leetcode.cn/problems/search-a-2d-matrix-ii/description/
 *
 * algorithms
 * Medium (53.20%)
 * Likes:    1385
 * Dislikes: 0
 * Total Accepted:    407K
 * Total Submissions: 765K
 * Testcase Example:  '[[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]\n' +
  '5'
 *
 * 编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：
 * 
 * 
 * 每行的元素从左到右升序排列。
 * 每列的元素从上到下升序排列。
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：matrix =
 * [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]],
 * target = 5
 * 输出：true
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：matrix =
 * [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]],
 * target = 20
 * 输出：false
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * m == matrix.length
 * n == matrix[i].length
 * 1 <= n, m <= 300
 * -10^9 <= matrix[i][j] <= 10^9
 * 每行的所有元素从左到右升序排列
 * 每列的所有元素从上到下升序排列
 * -10^9 <= target <= 10^9
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.暴力查找
    bool searchMatrix1(vector<vector<int>>& matrix, int target) {
        for (const auto& row : matrix) {
            for (auto& elem : row) {
                if (elem == target) {
                    return true;
                }
            }
        }

        return false;
    }

    // 2.二分查找, 遍历行/列，然后再对列/行进行二分
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        for (const auto& row : matrix) {
            int left = 0;
            int right = row.size() - 1;
            while (left < right) {
                // int mid = left + (right - left) / 2;
                int mid = left + right + 1 >> 1;
                if (row[mid] <= target) {
                    left = mid;
                } else {
                    right = mid - 1;
                }
            }
            if (row[right] == target) {
                return true;
            }
        }

        return false;
    }
};
// @lc code=end

