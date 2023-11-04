/*
 * @lc app=leetcode.cn id=151 lang=cpp
 *
 * [151] 反转字符串中的单词
 *
 * https://leetcode.cn/problems/reverse-words-in-a-string/description/
 *
 * algorithms
 * Medium (52.97%)
 * Likes:    1052
 * Dislikes: 0
 * Total Accepted:    468.9K
 * Total Submissions: 882.9K
 * Testcase Example:  '"the sky is blue"'
 *
 * 给你一个字符串 s ，请你反转字符串中 单词 的顺序。
 * 
 * 单词 是由非空格字符组成的字符串。s 中使用至少一个空格将字符串中的 单词 分隔开。
 * 
 * 返回 单词 顺序颠倒且 单词 之间用单个空格连接的结果字符串。
 * 
 * 注意：输入字符串 s中可能会存在前导空格、尾随空格或者单词间的多个空格。返回的结果字符串中，单词间应当仅用单个空格分隔，且不包含任何额外的空格。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：s = "the sky is blue"
 * 输出："blue is sky the"
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：s = "  hello world  "
 * 输出："world hello"
 * 解释：反转后的字符串中不能存在前导空格和尾随空格。
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：s = "a good   example"
 * 输出："example good a"
 * 解释：如果两个单词间有多余的空格，反转后的字符串需要将单词间的空格减少到仅有一个。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= s.length <= 10^4
 * s 包含英文大小写字母、数字和空格 ' '
 * s 中 至少存在一个 单词
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 进阶：如果字符串在你使用的编程语言中是一种可变数据类型，请尝试使用 O(1) 额外空间复杂度的 原地 解法。
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
                // reslut为空的时候，特殊处理
                if (result.size() == 0) {
                    result = word;
                } else {
                    result = word +  " " + result;;
                }
                
                word = "";
            } else if (ch != ' ') {
                word += ch;
            }
            left++;
        }
        // reslut为空的时候，特殊处理
        if (result.size() == 0) {
            result = word;
        } else {
            result = word +  " " + result;;
        }

        return result;
    }
};
// @lc code=end

