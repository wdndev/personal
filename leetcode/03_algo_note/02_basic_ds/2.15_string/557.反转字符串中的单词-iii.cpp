/*
 * @lc app=leetcode.cn id=557 lang=cpp
 *
 * [557] 反转字符串中的单词 III
 *
 * https://leetcode.cn/problems/reverse-words-in-a-string-iii/description/
 *
 * algorithms
 * Easy (73.67%)
 * Likes:    577
 * Dislikes: 0
 * Total Accepted:    318.8K
 * Total Submissions: 432.6K
 * Testcase Example:  `"Let's take LeetCode contest"`
 *
 * 给定一个字符串 s ，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词的初始顺序。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：s = "Let's take LeetCode contest"
 * 输出："s'teL ekat edoCteeL tsetnoc"
 * 
 * 
 * 示例 2:
 * 
 * 
 * 输入： s = "Mr Ding"
 * 输出："rM gniD"
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= s.length <= 5 * 10^4
 * s 包含可打印的 ASCII 字符。
 * s 不包含任何开头或结尾空格。
 * s 里 至少 有一个词。
 * s 中的所有单词都用一个空格隔开。
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    string reverseWords(string s) {
        if (s.empty()) {
            return s;
        }

        int start = 0;
        int end = s.size() - 1;

        // 去除字符串前后的空格
        while (start < s.size() && s[start] == ' ') {
            start++;
        }
        while (start < s.size() && s[end] == ' ') {
            end--;
        }
        // 切分出来的单词
        std::string word;
        // 结果
        std::string ans;

        // 开始遍历字符串，切分单词
        while (start <= end) {
            char ch = s[start];
            // 如果单词不为空，且ch字符串为空，则到了下一个单词，开始处理
            if (!word.empty() && ch == ' ') {
                this->revserse_str(word);
                ans = ans + word + " ";
                word = "";
            } else if (ch != ' ') {
                word += s[start];
            }

            start++;
        }
        // 处理最后一个单词
        this->revserse_str(word);
        ans = ans + word;
        
        return ans;
    }

private:
    void revserse_str(std::string& str) {
        for (int start = 0, end = str.size() - 1; start < end; start++, end--) {
            char tmp = str[start];
            str[start] = str[end];
            str[end] = tmp;
        }
    }
};
// @lc code=end

