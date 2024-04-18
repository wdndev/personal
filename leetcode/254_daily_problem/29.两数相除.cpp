// @before-stub-for-debug-begin
#include <vector>
#include <string>
#include "commoncppproblem29.h"

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=29 lang=cpp
 *
 * [29] 两数相除
 *
 * https://leetcode.cn/problems/divide-two-integers/description/
 *
 * algorithms
 * Medium (22.30%)
 * Likes:    1224
 * Dislikes: 0
 * Total Accepted:    235.7K
 * Total Submissions: 1.1M
 * Testcase Example:  '10\n3'
 *
 * 给你两个整数，被除数 dividend 和除数 divisor。将两数相除，要求 不使用 乘法、除法和取余运算。
 * 
 * 整数除法应该向零截断，也就是截去（truncate）其小数部分。例如，8.345 将被截断为 8 ，-2.7335 将被截断至 -2 。
 * 
 * 返回被除数 dividend 除以除数 divisor 得到的 商 。
 * 
 * 注意：假设我们的环境只能存储 32 位 有符号整数，其数值范围是 [−2^31,  2^31 − 1] 。本题中，如果商 严格大于 2^31 − 1
 * ，则返回 2^31 − 1 ；如果商 严格小于 -2^31 ，则返回 -2^31^ 。
 * 
 * 
 * 
 * 示例 1:
 * 
 * 
 * 输入: dividend = 10, divisor = 3
 * 输出: 3
 * 解释: 10/3 = 3.33333.. ，向零截断后得到 3 。
 * 
 * 示例 2:
 * 
 * 
 * 输入: dividend = 7, divisor = -3
 * 输出: -2
 * 解释: 7/-3 = -2.33333.. ，向零截断后得到 -2 。
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * -2^31 <= dividend, divisor <= 2^31 - 1
 * divisor != 0
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    int divide(int dividend, int divisor) {
        // 32 位最大：2147483647
        // 32 位最小： -2147483648
        // 边界注意会越界
        if (dividend == INT_MIN && divisor == -1) {
            return INT_MAX;
        }
        int sign = 1;
        if ((dividend > 0 && divisor < 0) || (dividend < 0 && divisor > 0)) {
            sign = -1;
        }
        // 防止越界，转为负数
        dividend = dividend > 0 ? -dividend : dividend;
        divisor = divisor > 0 ? -divisor : divisor;

        int ans = 0;
        while (dividend <= divisor) {
            int value = divisor;
            int k = 1;
            // 0xc0000000 是十进制 -2^30 的十六进制的表示
            // 判断 value >= 0xc0000000 的原因：保证 value + value 不会溢出
            // 可以这样判断的原因是：0xc0000000 是最小值 -2^31 的一半，
            // 而 a 的值不可能比 -2^31 还要小，所以 value 不可能比 0xc0000000 小
            while (value >= 0xc0000000 && dividend <= value + value) {
                value += value;
                // 代码优化：如果 k 已经大于最大值的一半的话，那么直接返回最小值
                // 因为这个时候 k += k 的话肯定会大于等于 2147483648 ，这个超过了题目给的范围
                if (k > INT_MAX / 2) {
                    return INT_MIN;
                }
                k += k;
            }
            dividend -= value;
            ans += k;
        }

        return sign == 1 ? ans : -ans;
    }
};
// @lc code=end

