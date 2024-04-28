/*
 * @lc app=leetcode.cn id=1017 lang=cpp
 *
 * [1017] 负二进制转换
 *
 * https://leetcode.cn/problems/convert-to-base-2/description/
 *
 * algorithms
 * Medium (64.95%)
 * Likes:    217
 * Dislikes: 0
 * Total Accepted:    34.2K
 * Total Submissions: 51.8K
 * Testcase Example:  '2'
 *
 * 给你一个整数 n ，以二进制字符串的形式返回该整数的 负二进制（base -2）表示。
 * 
 * 注意，除非字符串就是 "0"，否则返回的字符串中不能含有前导零。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：n = 2
 * 输出："110"
 * 解释：(-2)^2 + (-2)^1 = 2
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：n = 3
 * 输出："111"
 * 解释：(-2)^2 + (-2)^1 + (-2)^0 = 3
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：n = 4
 * 输出："100"
 * 解释：(-2)^2 = 4
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 0 <= n <= 10^9
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 对于转换一个整数到负二进制的表示，关键是处理基数为负数的除法和余数。
    // 在处理时，特别注意当除以 -2 时，余数需要非负。
    // 如果得到负余数，则需要调整商和余数，使余数非负。调整的方式即余数取绝对值, 商加 1
    // 如 -3 / -2 = 1 余 -1
    // 调整新的商为 2,余数为 1
    // 通过这种方式，我们确保了余数总是非负的，而调整后的商仍然保持了数值的平衡
    string baseNeg2(int n) {
        if (n == 0) {
            return "0";
        }
        std::string ans;
        while (n != 0) {
            // 计算余数
            int remainder = n % -2;
            n /= -2;

            // 如果余数为负数，则需要调整
            if (remainder < 0) {
                remainder += 2; // 使余数变为正
                n += 1;          // 调整商
            }
            ans.push_back(remainder + '0');
        }

        // 反转字符串，因为添加时是从最低位开始的
        reverse(ans.begin(), ans.end());
        return ans;
    }
};
// @lc code=end

