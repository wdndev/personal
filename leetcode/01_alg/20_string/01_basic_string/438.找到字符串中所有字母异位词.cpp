/*
 * @lc app=leetcode.cn id=438 lang=cpp
 *
 * [438] 找到字符串中所有字母异位词
 *
 * https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/
 *
 * algorithms
 * Medium (54.25%)
 * Likes:    1321
 * Dislikes: 0
 * Total Accepted:    339.8K
 * Total Submissions: 626.9K
 * Testcase Example:  '"cbaebabacd"\n"abc"'
 *
 * 给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。
 * 
 * 异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。
 * 
 * 
 * 
 * 示例 1:
 * 
 * 
 * 输入: s = "cbaebabacd", p = "abc"
 * 输出: [0,6]
 * 解释:
 * 起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
 * 起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。
 * 
 * 
 * 示例 2:
 * 
 * 
 * 输入: s = "abab", p = "ab"
 * 输出: [0,1,2]
 * 解释:
 * 起始索引等于 0 的子串是 "ab", 它是 "ab" 的异位词。
 * 起始索引等于 1 的子串是 "ba", 它是 "ab" 的异位词。
 * 起始索引等于 2 的子串是 "ab", 它是 "ab" 的异位词。
 * 
 * 
 * 
 * 
 * 提示:
 * 
 * 
 * 1 <= s.length, p.length <= 3 * 10^4
 * s 和 p 仅包含小写字母
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 构造两个相同的滑动窗口，在滑动窗口中维护每个字母的数量
    // 如果字母数量相同，则说明是 异位词
    vector<int> findAnagrams(string s, string p) {
        int s_len = s.size();
        int p_len = p.size();

        // 如果s的长度小于p的，则直接返回
        if (s_len < p_len) {
            return std::vector<int>();
        }

        // 结果
        std::vector<int> ans;
        // s字符串字母统计
        std::vector<int> s_count(26);
        // p字符串字母统计
        std::vector<int> p_count(26);

        // 首先统计前p_len个字符串中的字符数量
        for (int i = 0; i < p_len; i++) {
            s_count[s[i] - 'a']++;
            p_count[p[i] - 'a']++;
        }

        // 最开始如果相等，加入0
        if (s_count == p_count) {
            ans.emplace_back(0);
        }

        // 再遍历s中剩余的字符串
        for (int i = 0; i < s_len - p_len; i++) {
            s_count[s[i] - 'a']--;
            s_count[s[i + p_len] - 'a']++;

            if (s_count == p_count) {
                ans.emplace_back(i + 1);
            }
        }

        return ans;
    }
};
// @lc code=end

