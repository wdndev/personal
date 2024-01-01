/*
 * @lc app=leetcode.cn id=3 lang=cpp
 *
 * [3] 无重复字符的最长子串
 *
 * https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/
 *
 * algorithms
 * Medium (39.35%)
 * Likes:    9882
 * Dislikes: 0
 * Total Accepted:    2.6M
 * Total Submissions: 6.7M
 * Testcase Example:  '"abcabcbb"'
 *
 * 给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
 * 
 * 
 * 
 * 示例 1:
 * 
 * 
 * 输入: s = "abcabcbb"
 * 输出: 3 
 * 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
 * 
 * 
 * 示例 2:
 * 
 * 
 * 输入: s = "bbbbb"
 * 输出: 1
 * 解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
 * 
 * 
 * 示例 3:
 * 
 * 
 * 输入: s = "pwwkew"
 * 输出: 3
 * 解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
 * 请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 0 <= s.length <= 5 * 10^4
 * s 由英文字母、数字、符号和空格组成
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 滑动窗口 + hash
    int lengthOfLongestSubstring(string s) {
        // 哈希集合，记录每个字符是否出现过
        std::unordered_set<char> occ;
        int n = s.size();
        // 初始化右指针，初始值为-1，还没开始移动
        int right = -1;
        // 最长子串的长度
        int max_len = 0;

        // 开始遍历左指针位置
        for (int left = 0; left < n; left++) {
            if (left != 0) {
                // 左指针向右移动一格，一处一个字符
                occ.erase(s[left - 1]);
            }

            // 不断移动右指针，直到出现重复的字符，或是到达最后一个字符
            while (right + 1 < n && !occ.count(s[right + 1])) {
                occ.insert(s[right + 1]);
                right++;
            }

            // 更新max_len
            max_len = std::max(max_len, right - left + 1);
        }

        return max_len;
    }
};
// @lc code=end

