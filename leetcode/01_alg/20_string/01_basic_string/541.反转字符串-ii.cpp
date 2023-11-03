/*
 * @lc app=leetcode.cn id=541 lang=cpp
 *
 * [541] 反转字符串 II
 *
 * https://leetcode.cn/problems/reverse-string-ii/description/
 *
 * algorithms
 * Easy (57.70%)
 * Likes:    546
 * Dislikes: 0
 * Total Accepted:    234.9K
 * Total Submissions: 407.5K
 * Testcase Example:  '"abcdefg"\n2'
 *
 * 给定一个字符串 s 和一个整数 k，从字符串开头算起，每计数至 2k 个字符，就反转这 2k 字符中的前 k 个字符。
 * 
 * 
 * 如果剩余字符少于 k 个，则将剩余字符全部反转。
 * 如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：s = "abcdefg", k = 2
 * 输出："bacdfeg"
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：s = "abcd", k = 2
 * 输出："bacd"
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= s.length <= 10^4
 * s 仅由小写英文组成
 * 1 <= k <= 10^4
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    string reverseStr(string s, int k) {
        if (s.size() == 0) {
            return "";
        }
        if (k == 0) {
            return s;
        }

        for (int i = 0; i < s.size(); i += 2 * k) {
            int end = std::min(i + k - 1, int(s.size() - 1));
            this->reverse_str(s, i, end);
        }

        return s;
    }

    void reverse_str(string& s, int start, int end) {
        // 双指针
        while (start < end) {
            char tmp = s[start];
            s[start] = s[end];
            s[end] = tmp;

            start++;
            end--;
        }
    }
};
// @lc code=end

