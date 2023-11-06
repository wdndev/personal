/*
 * @lc app=leetcode.cn id=10 lang=cpp
 *
 * [10] 正则表达式匹配
 *
 * https://leetcode.cn/problems/regular-expression-matching/description/
 *
 * algorithms
 * Hard (30.71%)
 * Likes:    3775
 * Dislikes: 0
 * Total Accepted:    393.5K
 * Total Submissions: 1.3M
 * Testcase Example:  '"aa"\n"a"'
 *
 * 给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。
 * 
 * 
 * '.' 匹配任意单个字符
 * '*' 匹配零个或多个前面的那一个元素
 * 
 * 
 * 所谓匹配，是要涵盖 整个 字符串 s的，而不是部分字符串。
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：s = "aa", p = "a"
 * 输出：false
 * 解释："a" 无法匹配 "aa" 整个字符串。
 * 
 * 
 * 示例 2:
 * 
 * 
 * 输入：s = "aa", p = "a*"
 * 输出：true
 * 解释：因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：s = "ab", p = ".*"
 * 输出：true
 * 解释：".*" 表示可匹配零个或多个（'*'）任意字符（'.'）。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= s.length <= 20
 * 1 <= p.length <= 20
 * s 只包含从 a-z 的小写字母。
 * p 只包含从 a-z 的小写字母，以及字符 . 和 *。
 * 保证每次出现字符 * 时，前面都匹配到有效的字符
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    bool isMatch(string s, string p) {
        // 大小+1的目的是因为memo有边界限制
        m_memo = std::vector<std::vector<int>>(s.size() + 1, std::vector<int>(p.size() + 1, -1));
        return this->recur(s, p, 0, 0);
    }

    bool recur(std::string& s, std::string& p, int s_idx, int p_idx) {
        // 递归出口
        // 当 s_idx == s.size()，且 p_idx < p.size()
        // 可能p中还有*通配符
        if (p_idx == p.size()) {
            return s_idx == s.size();
        }

        // 如果在memo中有存储，直接返回结果
        if (m_memo[s_idx][p_idx] != -1) {
            return m_memo[s_idx][p_idx];
        }
        // 整个结果是否匹配
        bool res = false;
        // 当前第一个字符是否匹配
        bool first_match = false;

        // 处理 . 通配符
        if (s_idx < s.size()) {
            if(s[s_idx] == p[p_idx] || p[p_idx] == '.') {
                first_match = true;
            }
        }

        // 处理 * 通配符
        if ((p_idx + 1) < p.size() && p[p_idx + 1] == '*') {
            // 考虑只需两种情况：
            // 情况1：当前字符出现0次：跳过pattern中的当前字符和下一个"*"==>helper(s, p, si, pi + 2)
            // 情况2：当前字符出现1次：当前是否匹配 && 将字符s向后移动一位是否匹配==>cur_match && helper(s, p, si + 1, pi)

            res = this->recur(s, p, s_idx, p_idx + 2) || (first_match && this->recur(s, p, s_idx + 1, p_idx));
        } else {
            // //下一个不是"*"正常向后匹配就好
            res = first_match && this->recur(s, p, s_idx + 1, p_idx + 1);
        }

        m_memo[s_idx][p_idx] = res;

        return res;
    }

private:
    std::vector<std::vector<int>> m_memo;
};
// @lc code=end

