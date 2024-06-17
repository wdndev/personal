/*
 * @lc app=leetcode.cn id=522 lang=cpp
 *
 * [522] 最长特殊序列 II
 *
 * https://leetcode.cn/problems/longest-uncommon-subsequence-ii/description/
 *
 * algorithms
 * Medium (48.85%)
 * Likes:    233
 * Dislikes: 0
 * Total Accepted:    46.8K
 * Total Submissions: 91.5K
 * Testcase Example:  '["aba","cdc","eae"]'
 *
 * 给定字符串列表 strs ，返回其中 最长的特殊序列 的长度。如果最长特殊序列不存在，返回 -1 。
 * 
 * 特殊序列 定义如下：该序列为某字符串 独有的子序列（即不能是其他字符串的子序列）。
 * 
 * s 的 子序列可以通过删去字符串 s 中的某些字符实现。
 * 
 * 
 * 例如，"abc" 是 "aebdc" 的子序列，因为您可以删除"aebdc"中的下划线字符来得到 "abc"
 * 。"aebdc"的子序列还包括"aebdc"、 "aeb" 和 "" (空字符串)。
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入: strs = ["aba","cdc","eae"]
 * 输出: 3
 * 
 * 
 * 示例 2:
 * 
 * 
 * 输入: strs = ["aaa","aaa","aa"]
 * 输出: -1
 * 
 * 
 * 
 * 
 * 提示:
 * 
 * 
 * 2 <= strs.length <= 50
 * 1 <= strs[i].length <= 10
 * strs[i] 只包含小写英文字母
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 枚举每个字符串
    // 
    int findLUSlength(vector<string>& strs) {
        int n = strs.size();
        int ans = -1;
        for (int i = 0; i < n; i++) {
            bool check = true;
            for (int j = 0; j < n; j++) {
                if (i != j && this->is_subseq(strs[i], strs[j])) {
                    check = false;
                    break;
                }
            }
            if (check) {
                ans = std::max(ans, static_cast<int>(strs[i].size()));
            }
        }
        return ans;
    }
private:
    // 判断 s 是不是 t 的子串
    bool is_subseq(const std::string& s, const std::string& t) {
        std::cout << s << std::endl;
        int s_len = s.size();
        int t_len = t.size();
        int s_idx = 0;
        int t_idx = 0;
        // s指针指向s开始，t指针指向t开始
        // 如果s和t字符相同，s，t指针同时向后移动
        // 否则，s指针不动，t指针向后移动
        while(s_idx < s_len && t_idx < t_len)
        {
            if (s[s_idx] == t[t_idx])
            {
                s_idx++;
            }
            t_idx++;
        }

        return s_idx == s_len;
    }
};
// @lc code=end

