/*
 * @lc app=leetcode.cn id=680 lang=cpp
 *
 * [680] 验证回文串 II
 *
 * https://leetcode.cn/problems/valid-palindrome-ii/description/
 *
 * algorithms
 * Easy (40.07%)
 * Likes:    620
 * Dislikes: 0
 * Total Accepted:    141.3K
 * Total Submissions: 352.8K
 * Testcase Example:  '"aba"'
 *
 * 给你一个字符串 s，最多 可以从中删除一个字符。
 * 
 * 请你判断 s 是否能成为回文字符串：如果能，返回 true ；否则，返回 false 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：s = "aba"
 * 输出：true
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：s = "abca"
 * 输出：true
 * 解释：你可以删除字符 'c' 。
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：s = "abc"
 * 输出：false
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= s.length <= 10^5
 * s 由小写英文字母组成
 * 
 * 
 */

// @lc code=start
// 在允许最多删除一个字符的情况下，同样可以使用双指针，通过贪心实现。
// 初始化两个指针 left 和 right 分别指向字符串的第一个字符和最后一个字符。
// 每次判断两个指针指向的字符是否相同，如果相同，则更新指针，将left + 1，right - 1，
// 然后判断更新后的指针范围内的子串是否是回文字符串。

// 如果两个指针指向的字符不同，则两个字符中必须有一个被删除，此时就分成两种情况：

// - 删除左指针对应的字符，留下子串 s[left+1 : right]
// - 删除右指针对应的字符，留下子串 s[left : right−1]

// 当这两个子串中至少有一个是回文串时，就说明原始字符串删除一个字符之后就以成为回文串。
class Solution {
public:
    bool validPalindrome(string s) {
        if (s.size() == 0) {
            return true;
        }
        int left = 0;
        int right = s.size() - 1;

        while (left < right) {
            if (s[left] == s[right]) {
                left++;
                right--;
            } else {
                return this->check_palindrome(s, left + 1, right)  || this->check_palindrome(s, left, right - 1);
            }
        }

        return true;
    }

    // 检查字符串子串是不是回文串
    bool check_palindrome(std::string s, int left, int right) {
        for (int i = left, j = right; i < j; i++, j--) {
            if (s[i] != s[j]) {
                return false;
            }
        }

        return true;
    }
};
// @lc code=end

