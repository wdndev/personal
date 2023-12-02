/*
 * @lc app=leetcode.cn id=131 lang=cpp
 *
 * [131] 分割回文串
 *
 * https://leetcode.cn/problems/palindrome-partitioning/description/
 *
 * algorithms
 * Medium (73.40%)
 * Likes:    1692
 * Dislikes: 0
 * Total Accepted:    342.1K
 * Total Submissions: 466.1K
 * Testcase Example:  '"aab"'
 *
 * 给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是 回文串 。返回 s 所有可能的分割方案。
 * 
 * 回文串 是正着读和反着读都一样的字符串。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：s = "aab"
 * 输出：[["a","a","b"],["aa","b"]]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：s = "a"
 * 输出：[["a"]]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 
 * s 仅由小写英文字母组成
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    vector<vector<string>> partition(string s) {
        std::vector<std::vector<std::string>> ans;
        std::vector<std::string> tmp_str;

        this->dfs(0, s, tmp_str, ans);

        return ans;
    }

private:
    void dfs(int i, const std::string& s, std::vector<std::string> tmp_str, 
        std::vector<std::vector<std::string>>& ans) {
        if (i == s.size()) {
            ans.push_back(tmp_str);
            return;
        }

        for (int j = i; j < s.size(); j++) {
            if (this->is_palindrome(s, i, j)) {
                tmp_str.push_back(s.substr(i, j - i + 1));
                this->dfs(j + 1, s, tmp_str, ans);
                tmp_str.pop_back();
            }
        }
    }

    // 判断是否为回文字符串
    bool is_palindrome(const std::string& s, int start, int end) {
        while (start <= end) {
            if (s[start] != s[end]) {
                return false;
            }
            start++;
            end--;
        } 
        return true;
    }
};
// @lc code=end

