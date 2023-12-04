/*
 * @lc app=leetcode.cn id=20 lang=cpp
 *
 * [20] 有效的括号
 *
 * https://leetcode.cn/problems/valid-parentheses/description/
 *
 * algorithms
 * Easy (43.91%)
 * Likes:    4214
 * Dislikes: 0
 * Total Accepted:    1.6M
 * Total Submissions: 3.7M
 * Testcase Example:  '"()"'
 *
 * 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。
 * 
 * 有效字符串需满足：
 * 
 * 
 * 左括号必须用相同类型的右括号闭合。
 * 左括号必须以正确的顺序闭合。
 * 每个右括号都有一个对应的相同类型的左括号。
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：s = "()"
 * 输出：true
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：s = "()[]{}"
 * 输出：true
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：s = "(]"
 * 输出：false
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= s.length <= 10^4
 * s 仅由括号 '()[]{}' 组成
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    bool isValid(string s) {
        std::stack<char> stk;

        for (int i = 0; i < s.size(); i++) {
            // 左括号，入栈
            if (s[i] == '(' || s[i] == '[' || s[i] == '{') {
                stk.push(s[i]);
            } else {
            // 右括号处理
                // 如果栈为空，则直接返回
                if (stk.empty()) {
                    return false;
                }

                // 出栈，比对括号
                char tmp_char = stk.top();
                stk.pop();
                if (s[i] == ')' && tmp_char != '(') {
                    return false;
                }
                if (s[i] == ']' && tmp_char != '[') {
                    return false;
                }
                if (s[i] == '}' && tmp_char != '{') {
                    return false;
                }

            }
        }

        // 当所有元素遍历结束，栈空，则说明匹配成功
        return stk.empty();
    }
};
// @lc code=end

