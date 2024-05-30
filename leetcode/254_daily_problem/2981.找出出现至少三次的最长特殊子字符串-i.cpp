/*
 * @lc app=leetcode.cn id=2981 lang=cpp
 *
 * [2981] 找出出现至少三次的最长特殊子字符串 I
 *
 * https://leetcode.cn/problems/find-longest-special-substring-that-occurs-thrice-i/description/
 *
 * algorithms
 * Medium (48.81%)
 * Likes:    46
 * Dislikes: 0
 * Total Accepted:    20.5K
 * Total Submissions: 34.9K
 * Testcase Example:  '"aaaa"'
 *
 * 给你一个仅由小写英文字母组成的字符串 s 。
 * 
 * 如果一个字符串仅由单一字符组成，那么它被称为 特殊 字符串。例如，字符串 "abc" 不是特殊字符串，而字符串 "ddd"、"zz" 和 "f"
 * 是特殊字符串。
 * 
 * 返回在 s 中出现 至少三次 的 最长特殊子字符串 的长度，如果不存在出现至少三次的特殊子字符串，则返回 -1 。
 * 
 * 子字符串 是字符串中的一个连续 非空 字符序列。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：s = "aaaa"
 * 输出：2
 * 解释：出现三次的最长特殊子字符串是 "aa" ：子字符串 "aaaa"、"aaaa" 和 "aaaa"。
 * 可以证明最大长度是 2 。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：s = "abcdef"
 * 输出：-1
 * 解释：不存在出现至少三次的特殊子字符串。因此返回 -1 。
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：s = "abcaba"
 * 输出：1
 * 解释：出现三次的最长特殊子字符串是 "a" ：子字符串 "abcaba"、"abcaba" 和 "abcaba"。
 * 可以证明最大长度是 1 。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 3 <= s.length <= 50
 * s 仅由小写英文字母组成。
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 分类讨论
    // 由于特殊子串只包含单一字母，可以按照字母分组，每组统计相同字母连续出现的长度。
    // 如aaaabbbabb，可以分成4组，aaaa, bbb, a, bbb，则字母a的长度 [4,1], 字母b的长度[3, 2]
    // 遍历每个字母对应的长度列别a，把a从大到小排序
    // 取出子串的方法：
    // - 从最长特殊子串(a[0]) 中取长度均为 a[0]-2 的特殊子串。例如 aaaa 可以取三个aa
    // - 从最长和次长的特殊子串(a[0], a[1])中取三个长度一样的特殊子串
    //      - 如果 a[0] = a[1], 可以取长度均为 a[0] - 1的特殊子串
    //      - 如果 a[0] > a[1], 取长度为a[1]的特殊子串；a[0]取两个，a[1]一个
    //      - 合并：min(a[0] - 1, a[1])
    // - 从最长，次长，第三长的特殊子串(a[0], a[1], a[2])中各取一个长为a[2]的特殊子串
    // 取这三种的最大值， 即 max(a[0]-2. min(a[0]-1, a[1]), a[2])
    // 如果答案是0， 则返回-1
    // 代码实现时，在数组末尾加两个0，无法特判a的长度小于3的情况了
    int maximumLength(string s) {
        std::vector<int> groups[26];
        int cnt = 0;
        int n = s.size();
        for (int i = 0; i < n; i++) {
            cnt++;
            if (i + 1 == n || s[i] != s[i + 1]) {
                // 统计连续字符长度
                groups[s[i] - 'a'].push_back(cnt);
                cnt = 0;
            }
        }

        int ans = 0;
        
        for (auto& a: groups) {
            if (a.empty()) {
                continue;
            }
            std::sort(a.begin(), a.end(), std::greater());
            // 补两个0，无需判断 a 的长度小于3
            a.push_back(0);
            a.push_back(0);

            ans = std::max({ans, a[0] - 2, std::min(a[0] - 1, a[1]), a[2]});
        }

        return ans ? ans : -1;
        
    }
};
// @lc code=end

