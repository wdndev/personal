/*
 * @lc app=leetcode.cn id=14 lang=cpp
 *
 * [14] 最长公共前缀
 *
 * https://leetcode.cn/problems/longest-common-prefix/description/
 *
 * algorithms
 * Easy (43.54%)
 * Likes:    2974
 * Dislikes: 0
 * Total Accepted:    1.2M
 * Total Submissions: 2.7M
 * Testcase Example:  '["flower","flow","flight"]'
 *
 * 编写一个函数来查找字符串数组中的最长公共前缀。
 * 
 * 如果不存在公共前缀，返回空字符串 ""。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：strs = ["flower","flow","flight"]
 * 输出："fl"
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：strs = ["dog","racecar","car"]
 * 输出：""
 * 解释：输入不存在公共前缀。
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= strs.length <= 200
 * 0 <= strs[i].length <= 200
 * strs[i] 仅由小写英文字母组成
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        int word_num = strs.size();
        if (word_num == 0) {
            return "";
        }

        // 扫描第一个单词的字符
        for (int i = 0; i < strs[0].size(); i++) {
            // 取出第一个单词字符
            char ch = strs[0][i];
            // 依次扫面后续单词的字符
            // 依次和第一个单词的字符匹配
            for (int j = 1; j < word_num; j++) {
                if (i == strs[j].size() || ch != strs[j][i]) {
                    return strs[0].substr(0, i);
                }
            }
        }

        return strs[0];
    }
};
// @lc code=end

