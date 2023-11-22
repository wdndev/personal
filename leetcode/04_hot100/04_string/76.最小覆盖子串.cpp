/*
 * @lc app=leetcode.cn id=76 lang=cpp
 *
 * [76] 最小覆盖子串
 *
 * https://leetcode.cn/problems/minimum-window-substring/description/
 *
 * algorithms
 * Hard (45.42%)
 * Likes:    2750
 * Dislikes: 0
 * Total Accepted:    486.1K
 * Total Submissions: 1.1M
 * Testcase Example:  '"ADOBECODEBANC"\n"ABC"'
 *
 * 给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 ""
 * 。
 * 
 * 
 * 
 * 注意：
 * 
 * 
 * 对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。
 * 如果 s 中存在这样的子串，我们保证它是唯一的答案。
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：s = "ADOBECODEBANC", t = "ABC"
 * 输出："BANC"
 * 解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：s = "a", t = "a"
 * 输出："a"
 * 解释：整个字符串 s 是最小覆盖子串。
 * 
 * 
 * 示例 3:
 * 
 * 
 * 输入: s = "a", t = "aa"
 * 输出: ""
 * 解释: t 中两个字符 'a' 均应包含在 s 的子串中，
 * 因此没有符合条件的子字符串，返回空字符串。
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * ^m == s.length
 * ^n == t.length
 * 1 <= m, n <= 10^5
 * s 和 t 由英文字母组成
 * 
 * 
 * 
 * 进阶：你能设计一个在 o(m+n) 时间内解决此问题的算法吗？
 */

// @lc code=start
class Solution {
public:
    // 滑动窗口思想来解决，使用两个指针，right和left
    // right指针用来扩展滑动窗口，left指针用来收缩滑动窗口
    // 在任意时刻，只有一个指针运动，而另一个指针保持静止
    // 在s上滑动窗口，通过移动right指针不断扩展窗口，当滑动窗口中包含所有字符后，判断是否可以收缩，移动left
    // 使用哈希表表示字符串t中所有的字符及个数，
    // 用一个动态哈希表维护滑动窗口中所有的字符及个数
    // 如果这个动态哈希表中的所有自读，且对应的个数小于t的哈希表各个字符的个数，那么当前滑动窗口是可行的
    string minWindow(string s, string t) {
        // 将t中的字符，全部加入t hash表中
        for (auto& c : t) {
            ori_count[c]++;
        }

        // 滑动窗口左右定义
        int left = 0;
        int right = -1;

        // 结果字符串长度，及索引
        int ret_str_len = s.size() + 1;
        int ret_left = -1;
        // int ret_right = -1;

        // 如果滑动窗口右侧没有到s字符串的末尾，扩展
        while (right < int(s.size())) {
            // 判断滑动窗口右侧right移动后的字符是否在t的hash表中
            // 如果在，加入滑动窗口hash表
            if (ori_count.find(s[++right]) != ori_count.end() ) {
                dy_count[s[right]]++;
            }

            // 检查t中的字符，是否全部包含进滑动窗口的hash表中？
            // 如果包含，开始缩减滑动窗口大小
            // 否则，继续扩展滑动窗口
            while (check() && left <= right) {
                // 更新结果字符串的长度和坐标
                if (right - left + 1 < ret_str_len) {
                    ret_str_len = right - left + 1;
                    ret_left = left;
                }
                // 如果最左侧left字符不在t hash表中，则缩减滑动窗口
                if (ori_count.find(s[left]) != ori_count.end() ) {
                    dy_count[s[left]] --;
                }
                left++;
            }

        }

        return ret_left == -1 ? std::string() : s.substr(ret_left, ret_str_len);
    }
private:
    // t哈希表
    std::unordered_map<char, int> ori_count;
    // 滑动窗口hash表
    std::unordered_map<char, int> dy_count;

    // 判断t中的字符是否全包包含在滑动窗口的hash表中
    bool check() {
        for (auto& p : ori_count) {
            if (dy_count[p.first] < p.second) {
                return false;
            }
        }

        return true;
    }
};
// @lc code=end

