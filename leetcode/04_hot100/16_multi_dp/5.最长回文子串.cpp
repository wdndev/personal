/*
 * @lc app=leetcode.cn id=5 lang=cpp
 *
 * [5] 最长回文子串
 *
 * https://leetcode.cn/problems/longest-palindromic-substring/description/
 *
 * algorithms
 * Medium (37.90%)
 * Likes:    6966
 * Dislikes: 0
 * Total Accepted:    1.6M
 * Total Submissions: 4.1M
 * Testcase Example:  '"babad"'
 *
 * 给你一个字符串 s，找到 s 中最长的回文子串。
 * 
 * 如果字符串的反序与原始字符串相同，则该字符串称为回文字符串。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：s = "babad"
 * 输出："bab"
 * 解释："aba" 同样是符合题意的答案。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：s = "cbbd"
 * 输出："bb"
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= s.length <= 1000
 * s 仅由数字和英文字母组成
 * 
 * 
 */

// @lc code=start
// 1.暴力求解，列举所有的子串，判断是否为回文串
class Solution1 {
public:
    string longestPalindrome(string s) {
        if (s.size() <= 1) {
            return s;
        }
        std::string ans;
        int max_len = 0;
        
        for (int i = 0; i < s.size() - 1; i++) {
            for (int j = i + 1; j < s.size(); j++) {
                std::string tmp_str = s.substr(i, j - i + 1);
                if (this->check_palindrome(tmp_str) && tmp_str.size() > max_len) {
                    ans = tmp_str;
                    max_len = tmp_str.size();
                }
            }
        }

        return ans;
    }

    // 检查字符串子串是不是回文串
    bool check_palindrome(std::string& s) {
        for (int i = 0, j = s.size() - 1; i < j; i++, j--) {
            if (s[i] != s[j]) {
                return false;
            }
        }

        return true;
    }
};

// 2.暴力 + 动态规划
class Solution2 {
public:
    string longestPalindrome(string s) {
        int str_len = s.size();
        if (str_len < 2) {
            return s;
        }

        std::vector<std::vector<bool>> dp(str_len, std::vector<bool>(str_len));
        // 最长子串的开始位置和最大长度
        int max_len = 0;
        int begin = 0;

        // 遍历所有长度
        for (int len = 1; len <= str_len; len++) {
            // 枚举左边界
            for (int start = 0; start < str_len; start++) {
                // 根据左边界和长度，确定结束位置
                int end = start + len - 1;
                // 下标越界
                if (end >= str_len) {
                    break;
                }
                // dp[i][j] = dp[i + 1][j - 1] && s[i] == s[j]
                dp[start][end] = s[start] == s[end] && (len == 1 || len == 2 || dp[start + 1][end - 1]);

                if (dp[start][end] && len > max_len)  {
                    max_len = len;
                    begin = start;
                }
            }
        }

        return s.substr(begin, max_len);
    }
};

// 3.中心向外扩散
class Solution {
public:
    string longestPalindrome(string s) {
        int len = s.size();
        if (len < 2) {
            return s;
        }

        for (int i = 0; i < len - 1; i++) {
            // 奇数长度
            this->extend_palindrome(s, i, i);
            // 偶数长度
            this->extend_palindrome(s, i, i + 1);
        }

        return s.substr(m_start, m_max_len);
    }

    void extend_palindrome(std::string& s, int left, int right) {
        while (left >= 0 && right < s.size() && s[left] == s[right]) {
            left--;
            right++;
        }

        if (m_max_len < right - left - 1) {
            m_start = left + 1;
            m_max_len = right - left - 1;
        }
    }
private:
    int m_start;
    int m_max_len;
};
// @lc code=end

