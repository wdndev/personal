/*
 * @lc app=leetcode.cn id=567 lang=cpp
 *
 * [567] 字符串的排列
 *
 * https://leetcode.cn/problems/permutation-in-string/description/
 *
 * algorithms
 * Medium (44.68%)
 * Likes:    968
 * Dislikes: 0
 * Total Accepted:    275.4K
 * Total Submissions: 616.2K
 * Testcase Example:  '"ab"\n"eidbaooo"'
 *
 * 给你两个字符串 s1 和 s2 ，写一个函数来判断 s2 是否包含 s1 的排列。如果是，返回 true ；否则，返回 false 。
 * 
 * 换句话说，s1 的排列之一是 s2 的 子串 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：s1 = "ab" s2 = "eidbaooo"
 * 输出：true
 * 解释：s2 包含 s1 的排列之一 ("ba").
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：s1= "ab" s2 = "eidboaoo"
 * 输出：false
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= s1.length, s2.length <= 10^4
 * s1 和 s2 仅包含小写字母
 * 
 * 
 */

// @lc code=start
// 使用两个数组 s1_cnt 和 s2_cnt ， s1_cnt统计 s1中各个字符的个数，
// s2_cnt 统计当前遍历的子串中各个字符的个数。
// 由于需要遍历的子串长度均为 s1_len，可以使用一个固定长度为 s1_len 的滑动窗口来维护 s2_cnt：
// 滑动窗口每向右滑动一次，就多统计一次进入窗口的字符，少统计一次离开窗口的字符。
// 然后，判断 s1_cnt 是否与 s2_cnt 相等，若相等则意味着 s1的排列之一是 s2的子串。
class Solution {
public:
    bool checkInclusion(string s1, string s2) {
        int s1_len = s1.length();
        int s2_len = s2.length();

        if (s1_len > s2_len) {
            return false;
        }

        std::vector<int> s1_cnt(26);
        std::vector<int> s2_cnt(26);

        for (int i = 0; i < s1_len; i++) {
            s1_cnt[s1[i] - 'a']++;
            s2_cnt[s2[i] - 'a']++;
        }

        if (s1_cnt == s2_cnt) {
            return true;
        }

        for (int i = s1_len; i < s2_len; i++) {
            s2_cnt[s2[i] - 'a']++;
            s2_cnt[s2[i - s1_len] - 'a']--;

            if (s1_cnt == s2_cnt) {
                return true;
            }
        }

        return false;
    }
};
// @lc code=end

