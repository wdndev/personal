/*
 * @lc app=leetcode.cn id=557 lang=cpp
 *
 * [557] 反转字符串中的单词 III
 *
 * https://leetcode.cn/problems/reverse-words-in-a-string-iii/description/
 *
 * algorithms
 * Easy (73.70%)
 * Likes:    568
 * Dislikes: 0
 * Total Accepted:    315.1K
 * Total Submissions: 427.5K
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
 * 输入： s = "God Ding"
 * 输出："doG gniD"
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
        int str_len = s.size();
        if (str_len == 0) {
            return "";
        }

        int left = 0;
        int right = s.size() - 1;
        std::string result;
        std::string word;

        // 删除前面的空格
        while (left < str_len && s[left] == ' ') {
            left++;
        }

        // 删除后面的空格
        while (right < str_len && s[right] == ' ') {
            right--;
        }

        // 切分单词
        while (left <= right) {
            char ch = s[left];
            // 如果单词不为空，且ch字符为空，则到下一个单词了，开始处理
            if (word.size() != 0 && ch == ' ') {
                // 翻转单词
                this->resver_str(word);
                result = result + word +  " ";
                word = "";
            } else if (ch != ' ') {
                word += ch;
            }
            left++;
        }
        // 处理最后一个单词
        this->resver_str(word);
        result = result + word;

        return result;
    }

    void resver_str(std::string& word) {
        int low = 0;
        int high = word.size() - 1;
        while (low <= high) {
            char tmp = word[low];
            word[low] = word[high];
            word[high] = tmp;
            low++;
            high--;
        }
    }
};
// @lc code=end

