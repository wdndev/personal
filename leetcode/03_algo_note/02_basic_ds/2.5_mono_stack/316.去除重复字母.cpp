// @before-stub-for-debug-begin
#include <vector>
#include <string>
#include "commoncppproblem316.h"

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=316 lang=cpp
 *
 * [316] 去除重复字母
 *
 * https://leetcode.cn/problems/remove-duplicate-letters/description/
 *
 * algorithms
 * Medium (48.69%)
 * Likes:    1026
 * Dislikes: 0
 * Total Accepted:    128.9K
 * Total Submissions: 264.7K
 * Testcase Example:  '"bcabc"'
 *
 * 给你一个字符串 s ，请你去除字符串中重复的字母，使得每个字母只出现一次。需保证 返回结果的字典序最小（要求不能打乱其他字符的相对位置）。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：s = "bcabc"
 * 输出："abc"
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：s = "cbacdcbc"
 * 输出："acdb"
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= s.length <= 10^4
 * s 由小写英文字母组成
 * 
 * 
 * 
 * 
 * 注意：该题与 1081
 * https://leetcode-cn.com/problems/smallest-subsequence-of-distinct-characters
 * 相同
 * 
 */

// @lc code=start
class Solution {
public:
    string removeDuplicateLetters(string s) {
        std::vector<char> stk;
        std::unordered_map<char, int> letter_counts;
        // 统计各单词出现的频率
        for (auto& ch : s) {
            letter_counts[ch]++;
        }

        for (auto& ch : s) {
            if (!count(stk.begin(), stk.end(), ch)) {
                while (!stk.empty() && ch < stk.back() && letter_counts[stk.back()] > 0) {
                    stk.pop_back();
                }
                stk.push_back(ch);
            }
            letter_counts[ch]--;
        }

        std::string ans(stk.begin(), stk.end());

        return ans;
    }
};
// @lc code=end

