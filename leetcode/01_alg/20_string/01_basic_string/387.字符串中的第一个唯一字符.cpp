/*
 * @lc app=leetcode.cn id=387 lang=cpp
 *
 * [387] 字符串中的第一个唯一字符
 *
 * https://leetcode.cn/problems/first-unique-character-in-a-string/description/
 *
 * algorithms
 * Easy (56.04%)
 * Likes:    710
 * Dislikes: 0
 * Total Accepted:    401.4K
 * Total Submissions: 716K
 * Testcase Example:  '"leetcode"'
 *
 * 给定一个字符串 s ，找到 它的第一个不重复的字符，并返回它的索引 。如果不存在，则返回 -1 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入: s = "leetcode"
 * 输出: 0
 * 
 * 
 * 示例 2:
 * 
 * 
 * 输入: s = "loveleetcode"
 * 输出: 2
 * 
 * 
 * 示例 3:
 * 
 * 
 * 输入: s = "aabb"
 * 输出: -1
 * 
 * 
 * 
 * 
 * 提示:
 * 
 * 
 * 1 <= s.length <= 10^5
 * s 只包含小写字母
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    int firstUniqChar(string s) {
        std::unordered_map<int, int> str_map;

        for (auto& ch : s) {
            str_map[ch]++;
        }

        for (int i = 0; i < s.size(); i++) {
            char ch = s[i];
            if (str_map[ch] == 1) {
                return i;
            }
        }

        return -1;
    }
};
// @lc code=end

