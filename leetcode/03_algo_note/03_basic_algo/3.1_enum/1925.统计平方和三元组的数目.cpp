/*
 * @lc app=leetcode.cn id=1925 lang=cpp
 *
 * [1925] 统计平方和三元组的数目
 *
 * https://leetcode.cn/problems/count-square-sum-triples/description/
 *
 * algorithms
 * Easy (68.92%)
 * Likes:    21
 * Dislikes: 0
 * Total Accepted:    14K
 * Total Submissions: 20.3K
 * Testcase Example:  '5'
 *
 * 一个 平方和三元组 (a,b,c) 指的是满足 a^2 + b^2 = c^2 的 整数 三元组 a，b 和 c 。
 * 
 * 给你一个整数 n ，请你返回满足 1 <= a, b, c <= n 的 平方和三元组 的数目。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 输入：n = 5
 * 输出：2
 * 解释：平方和三元组为 (3,4,5) 和 (4,3,5) 。
 * 
 * 
 * 示例 2：
 * 
 * 输入：n = 10
 * 输出：4
 * 解释：平方和三元组为 (3,4,5)，(4,3,5)，(6,8,10) 和 (8,6,10) 。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= n <= 250
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.枚举算法
    int countTriples(int n) {
        int count = 0;
        for (int a = 1; a <= n; a++) {
            for (int b = 1; b <= n; b++) {
                // 在计算中，为了防止浮点数造成的误差，
                // 并且两个相邻的完全平方正数之间的距离一定大于 1，
                int c = sqrt(a * a + b * b + 1);
                if (c <= n && a * a + b * b == c * c) {
                    count++;
                }
            }
        }

        return count;
    }
};
// @lc code=end

