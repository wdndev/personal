/*
 * @lc app=leetcode.cn id=50 lang=cpp
 *
 * [50] Pow(x, n)
 *
 * https://leetcode.cn/problems/powx-n/description/
 *
 * algorithms
 * Medium (38.01%)
 * Likes:    1258
 * Dislikes: 0
 * Total Accepted:    404.6K
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
    // 1.暴力 O(n), 超时
    double myPow1(double x, int n) {
        double result = 1.0;
        if (n == 0 || x == 1) {
            return result;
        } 
        bool flag_negative = false;
        if (n < 0) {
            flag_negative = true;
            n = -n;
        }
        for (int i = 0; i < n; i++) {
            result *= x;
        }

        return flag_negative ? 1 / result : result;
    }

    // 2. 分治
    // template:
    //     1. terminator
    //     2. process (split your big problem)
    //     3. drill down (subproblems), merge(subresult)
    //     4. reverse states
    double myPow(double x, int n) {
        long long N = n;
        // 如果n是负数，特殊处理
        return N >= 0 ? fast_pow(x, N) : fast_pow(1/x, -N);
    }
private:
    // 分治，递归
    // 2^10 --> 2^5  -> (2^2) * 2
    double fast_pow(double x, int n) {
        if (n == 0)
            return 1.0;
        double sub_result = fast_pow(x, n/2);
        return n % 2 == 0 ? sub_result * sub_result : sub_result * sub_result * x;
    }
};
// @lc code=end

