/*
 * @lc app=leetcode.cn id=50 lang=cpp
 *
 * [50] Pow(x, n)
 *
 * https://leetcode.cn/problems/powx-n/description/
 *
 * algorithms
 * Medium (38.06%)
 * Likes:    1289
 * Dislikes: 0
 * Total Accepted:    414.2K
 * Total Submissions: 1.1M
 * Testcase Example:  '2.00000\n10'
 *
 * 实现 pow(x, n) ，即计算 x 的整数 n 次幂函数（即，x^n^ ）。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：x = 2.00000, n = 10
 * 输出：1024.00000
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：x = 2.10000, n = 3
 * 输出：9.26100
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：x = 2.00000, n = -2
 * 输出：0.25000
 * 解释：2^-2 = 1/2^2 = 1/4 = 0.25
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * -100.0 < x < 100.0
 * -2^31 <= n <= 2^31-1
 * n 是一个整数
 * 要么 x 不为零，要么 n > 0 。
 * -10^4 <= x^n <= 10^4
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 分治，递归
    // 2^10  -> 2^5 -> 2^2 * 2
    double myPow(double x, int n) {
        long long N = n;
        // 如果n是负数，特殊处理
        return N >=0 ? this->fast_pow(x, N) : this->fast_pow(1/x, -N);
    }

    double fast_pow(double x, int n) {
        if (n == 0) {
            return 1.0;
        }

        double sub_result = this->fast_pow(x, n / 2);
        // 奇偶数特殊处理
        return n % 2 == 0 ? sub_result * sub_result : x * sub_result * sub_result;
    }

};
// @lc code=end

