/*
 * @lc app=leetcode.cn id=1329 lang=cpp
 *
 * [1329] 将矩阵按对角线排序
 *
 * https://leetcode.cn/problems/sort-the-matrix-diagonally/description/
 *
 * algorithms
 * Medium (77.85%)
 * Likes:    123
 * Dislikes: 0
 * Total Accepted:    24.3K
 * Total Submissions: 30.2K
 * Testcase Example:  '[[3,3,1,1],[2,2,1,2],[1,1,1,2]]'
 *
 * 矩阵对角线 是一条从矩阵最上面行或者最左侧列中的某个元素开始的对角线，沿右下方向一直到矩阵末尾的元素。例如，矩阵 mat 有 6 行 3 列，从
 * mat[2][0] 开始的 矩阵对角线 将会经过 mat[2][0]、mat[3][1] 和 mat[4][2] 。
 * 
 * 给你一个 m * n 的整数矩阵 mat ，请你将同一条 矩阵对角线 上的元素按升序排序后，返回排好序的矩阵。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 
 * 
 * 输入：mat = [[3,3,1,1],[2,2,1,2],[1,1,1,2]]
 * 输出：[[1,1,1,1],[1,2,2,2],[1,2,3,3]]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：mat =
 * [[11,25,66,1,69,7],[23,55,17,45,15,52],[75,31,36,44,58,8],[22,27,33,25,68,4],[84,28,14,11,5,50]]
 * 
 * 输出：[[5,17,4,1,52,7],[11,11,25,45,8,69],[14,23,25,44,58,15],[22,27,31,36,50,66],[84,28,75,33,55,68]]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * m == mat.length
 * n == mat[i].length
 * 1 
 * 1 
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 对角线上的坐标满足性质：行下标减列下标等于一个定值 k=-1
    // 对角线，从右到左遍历，即 (0, 3) -> (0, 2)
    // 设坐标为 (i, j), 设 k = i - j
    // - 第一条对角线上只有一个点，坐标为 (0, n-1)，其 k = 1 - n
    // - 最后一条对角线上也只有一个点，坐标为 (m - 1, 0), 其 k = m - 1
    // - 所以枚举对角线，就是枚举 k 从 1-n到m-1
    // 对于同一条对角线，知道了下标i，就知道了下标j=i-k
    // - i的最小值：令j=0，则i=k,即i最小为man(k, 0)
    // - i的最大值：令j=n-1，则i=k+n-1，所以i的最大值为min(k+n-1, m-1)
    // - 枚举i，范围为左闭右开的区间 [max(k ,0), min(k+n, m))
    //  - 依次把对角线的元素加入一个数组中，从小到大排序后，在填入
    vector<vector<int>> diagonalSort(vector<vector<int>>& mat) {
        int m = mat.size();
        int n = mat[0].size();
        std::vector<int> tmp_vec(std::min(m, n));

        // 遍历k
        // 注意：k = i - j
        for (int k = 1 - n; k < m - 1; k++) {
            int left_i = std::max(k ,0);
            int right_i = std::min(k + n, m);
            for (int i = left_i; i < right_i; i++) {
                tmp_vec[i - left_i] = mat[i][i - k];
            }
            std::sort(tmp_vec.begin(), tmp_vec.begin() + (right_i - left_i));
            for (int i = left_i; i < right_i; i++) {
                mat[i][i - k] = tmp_vec[i - left_i];
            }
        }

        return mat;
    }
};
// @lc code=end

