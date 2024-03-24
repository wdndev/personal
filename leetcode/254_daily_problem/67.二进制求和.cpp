/*
 * @lc app=leetcode.cn id=67 lang=cpp
 *
 * [67] 二进制求和
 *
 * https://leetcode.cn/problems/add-binary/description/
 *
 * algorithms
 * Easy (52.99%)
 * Likes:    1179
 * Dislikes: 0
 * Total Accepted:    384.2K
 * Total Submissions: 725K
 * Testcase Example:  '"11"\n"1"'
 *
 * 给你两个二进制字符串 a 和 b ，以二进制字符串的形式返回它们的和。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入:a = "11", b = "1"
 * 输出："100"
 * 
 * 示例 2：
 * 
 * 
 * 输入：a = "1010", b = "1011"
 * 输出："10101"
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= a.length, b.length <= 10^4
 * a 和 b 仅由字符 '0' 或 '1' 组成
 * 字符串如果不是 "0" ，就不含前导零
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 可以借鉴「列竖式」的方法，末尾对齐，逐位相加。在十进制的计算中「逢十进一」，二进制中需要「逢二进一」。
    // 可以取 n=max{∣a∣,∣b∣}，循环 n 次，从最低位开始遍历。使用一个变量 carry 表示上一个位置的进位，初始值为 0。
    // 记当前位置对其的两个位为 ai和 bi，则每一位的答案为 (carry+ai+bi) mod 2，下一位的进位为 ⌊(carry+ai+bi)/2⌋。
    // 重复上述步骤，直到数字 a 和 b 的每一位计算完毕。最后如果 carry 的最高位不为 0，则将最高位添加到计算结果的末尾。

    string addBinary(string a, string b) {
        std::string ans;
        
        reverse(a.begin(), a.end());
        reverse(b.begin(), b.end());

        int n =std::max(a.size(), b.size());
        int carry = 0;

        for (size_t i = 0; i < n; i++) {
            carry += i < a.size() ? (a.at(i) == '1') : 0;
            carry += i < b.size() ? (b.at(i) == '1') : 0;
            ans.push_back((carry % 2) ? '1' : '0');
            carry /= 2;
        }

        if (carry) {
            ans.push_back('1');
        }

        reverse(ans.begin(), ans.end());

        return ans;
    }
};
// @lc code=end

