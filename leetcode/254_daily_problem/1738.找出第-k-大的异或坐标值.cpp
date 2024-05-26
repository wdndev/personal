/*
 * @lc app=leetcode.cn id=1738 lang=cpp
 *
 * [1738] 找出第 K 大的异或坐标值
 *
 * https://leetcode.cn/problems/find-kth-largest-xor-coordinate-value/description/
 *
 * algorithms
 * Medium (65.04%)
 * Likes:    119
 * Dislikes: 0
 * Total Accepted:    40.5K
 * Total Submissions: 60.4K
 * Testcase Example:  '[[5,2],[1,6]]\n1'
 *
 * 给你一个二维矩阵 matrix 和一个整数 k ，矩阵大小为 m x n 由非负整数组成。
 * 
 * 矩阵中坐标 (a, b) 的 值 可由对所有满足 0 <= i <= a < m 且 0 <= j <= b < n 的元素
 * matrix[i][j]（下标从 0 开始计数）执行异或运算得到。
 * 
 * 请你找出 matrix 的所有坐标中第 k 大的值（k 的值从 1 开始计数）。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 输入：matrix = [[5,2],[1,6]], k = 1
 * 输出：7
 * 解释：坐标 (0,1) 的值是 5 XOR 2 = 7 ，为最大的值。
 * 
 * 示例 2：
 * 
 * 输入：matrix = [[5,2],[1,6]], k = 2
 * 输出：5
 * 解释：坐标 (0,0) 的值是 5 = 5 ，为第 2 大的值。
 * 
 * 示例 3：
 * 
 * 输入：matrix = [[5,2],[1,6]], k = 3
 * 输出：4
 * 解释：坐标 (1,0) 的值是 5 XOR 1 = 4 ，为第 3 大的值。
 * 
 * 示例 4：
 * 
 * 输入：matrix = [[5,2],[1,6]], k = 4
 * 输出：0
 * 解释：坐标 (1,1) 的值是 5 XOR 2 XOR 1 XOR 6 = 0 ，为第 4 大的值。
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * m == matrix.length
 * n == matrix[i].length
 * 1 <= m, n <= 1000
 * 0 <= matrix[i][j] <= 10^6
 * 1 <= k <= m * n
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 二维前缀异或和
    // 类似二维前缀和，定义 s[i+1][j+1]表示左上角在 [0,0]，右下角在[i,j]的子矩阵异或和，则有：
    //  s[i+1][j+1] = s[i+1][j] ⊕ s[i][j+1] ⊕ s[i][j] ⊕ matrix[i][j]
    // 由于一个数异或自己等于0，s[i+1][j] ⊕ s[i][j+1]会导致 s[i][j] 这部分被抵消，所以要异或进来
    int kthLargestValue(vector<vector<int>>& matrix, int k) {
        int m = matrix.size();
        int n = matrix[0].size();
        std::vector<int> arr(m * n);
        std::vector<std::vector<int>> s(m + 1, std::vector<int>(n + 1));

        int idx = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                s[i + 1][j + 1] = s[i + 1][j] ^ s[i][j + 1] ^ s[i][j] ^ matrix[i][j];
                arr[idx] = s[i + 1][j + 1];
                idx++;
            }
        }

        std::sort(arr.begin(), arr.end());

        return arr[idx - k];
    }
};
// @lc code=end

