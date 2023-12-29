/*
 * @lc app=leetcode.cn id=345 lang=cpp
 *
 * [345] 反转字符串中的元音字母
 *
 * https://leetcode.cn/problems/reverse-vowels-of-a-string/description/
 *
 * algorithms
 * Easy (54.66%)
 * Likes:    340
 * Dislikes: 0
 * Total Accepted:    186.9K
 * Total Submissions: 341.8K
 * Testcase Example:  '"hello"'
 *
 * 给你一个字符串 s ，仅反转字符串中的所有元音字母，并返回结果字符串。
 * 
 * 元音字母包括 'a'、'e'、'i'、'o'、'u'，且可能以大小写两种形式出现不止一次。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：s = "hello"
 * 输出："holle"
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：s = "leetcode"
 * 输出："leotcede"
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= s.length <= 3 * 10^5
 * s 由 可打印的 ASCII 字符组成
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    string reverseVowels(string s) {
        int left = 0;
        int right = s.size() - 1;
        while (left < right) {
            while (left < right &&!isVowel(s[left])) {
                left++;
            }
            while (left < right &&!isVowel(s[right])) {
                right--;
            }
            swap(s[left], s[right]);
            left++;
            right--;
        }

        return s;
    }

    bool isVowel(char ch) {
        bool lower = ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u';
        bool upper = ch == 'A' || ch == 'E' || ch == 'I' || ch == 'O' || ch == 'U';
        return lower || upper;
    }
};
// @lc code=end

