/*
 * @lc app=leetcode.cn id=1763 lang=cpp
 *
 * [1763] 最长的美好子字符串
 *
 * https://leetcode.cn/problems/longest-nice-substring/description/
 *
 * algorithms
 * Easy (66.79%)
 * Likes:    229
 * Dislikes: 0
 * Total Accepted:    30.9K
 * Total Submissions: 46.4K
 * Testcase Example:  '"YazaAay"'
 *
 * 当一个字符串 s 包含的每一种字母的大写和小写形式 同时 出现在 s 中，就称这个字符串 s 是 美好 字符串。比方说，"abABB"
 * 是美好字符串，因为 'A' 和 'a' 同时出现了，且 'B' 和 'b' 也同时出现了。然而，"abA" 不是美好字符串因为 'b' 出现了，而
 * 'B' 没有出现。
 * 
 * 给你一个字符串 s ，请你返回 s 最长的 美好子字符串 。如果有多个答案，请你返回 最早
 * 出现的一个。如果不存在美好子字符串，请你返回一个空字符串。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：s = "YazaAay"
 * 输出："aAa"
 * 解释："aAa" 是一个美好字符串，因为这个子串中仅含一种字母，其小写形式 'a' 和大写形式 'A' 也同时出现了。
 * "aAa" 是最长的美好子字符串。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：s = "Bb"
 * 输出："Bb"
 * 解释："Bb" 是美好字符串，因为 'B' 和 'b' 都出现了。整个字符串也是原字符串的子字符串。
 * 
 * 示例 3：
 * 
 * 
 * 输入：s = "c"
 * 输出：""
 * 解释：没有美好子字符串。
 * 
 * 示例 4：
 * 
 * 
 * 输入：s = "dDzeE"
 * 输出："dD"
 * 解释："dD" 和 "eE" 都是最长美好子字符串。
 * 由于有多个美好子字符串，返回 "dD" ，因为它出现得最早。
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 
 * s 只包含大写和小写英文字母。
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.枚举字符串
    // 由于英文字母最多有26个，可以使用二进制位来标记，lower标记字符中出现过的小写，upper标记字符中出现过的大写
    // 如果lower = upper，则认为字符串中所有的字符都满足大小写形式同时出现
    // 则认定为美好字符串
    string longestNiceSubstring(string s) {
        int max_pos = 0;
        int max_len = 0;
        for (int i = 0; i < s.size(); i++) {
            int lower = 0;
            int upper = 0;
            for (int j = i; j < s.size(); j++) {
                if (islower(s[j])) {
                    lower |= 1 << (s[j] - 'a');
                } else {
                    upper |= 1 << (s[j] - 'A');
                }

                if (lower == upper && j - i + 1 > max_len) {
                    max_len = j - i + 1;
                    max_pos = i;
                }
            }
        }
        return s.substr(max_pos, max_len);
    }
};
// @lc code=end

