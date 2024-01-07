/*
 * @lc app=leetcode.cn id=394 lang=cpp
 *
 * [394] 字符串解码
 *
 * https://leetcode.cn/problems/decode-string/description/
 *
 * algorithms
 * Medium (57.13%)
 * Likes:    1658
 * Dislikes: 0
 * Total Accepted:    280.9K
 * Total Submissions: 491.6K
 * Testcase Example:  '"3[a]2[bc]"'
 *
 * 给定一个经过编码的字符串，返回它解码后的字符串。
 * 
 * 编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。
 * 
 * 你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。
 * 
 * 此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：s = "3[a]2[bc]"
 * 输出："aaabcbc"
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：s = "3[a2[c]]"
 * 输出："accaccacc"
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：s = "2[abc]3[cd]ef"
 * 输出："abcabccdcdcdef"
 * 
 * 
 * 示例 4：
 * 
 * 
 * 输入：s = "abc3[cd]xyz"
 * 输出："abccdcdcdxyz"
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= s.length <= 30
 * s 由小写英文字母、数字和方括号 '[]' 组成
 * s 保证是一个 有效 的输入。
 * s 中所有整数的取值范围为 [1, 300] 
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    string decodeString(string s) {
        std::stack<char> stk;
        std::string tmp_str = "";
        for (int i = 0; i < s.size(); i++) {
            if (s[i] != ']') {
                stk.push(s[i]);
            } else {
                // 取出字符串
                tmp_str = "";
                while (!stk.empty() && stk.top() != '[') {
                    tmp_str = tmp_str + stk.top();
                    stk.pop();
                }
                // 弹出 [
                stk.pop();

                // 取出数字
                int num = 0;    // 表示读取到的数字
                int n = 0;      // 表示第几位数字
                while (!stk.empty() && isdigit(stk.top())) {
                    int po = pow(10, n);
                    num = (stk.top() - '0') * po + num;
                    n++;

                    stk.pop();
                }

                // 把num个tmp_str放入栈中
                for (int j = 0; j < num; j++) {
                    for (int k = tmp_str.size() - 1; k >= 0; k--) {
                        stk.push(tmp_str[k]);
                    }
                }

            }
        }

        std::string ans;
        while (!stk.empty()) {
            char tmp_char = stk.top();
            stk.pop();
            ans = tmp_char + ans;
        }

        return ans;
    }
};

// @lc code=end

