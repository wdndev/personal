/*
 * @lc app=leetcode.cn id=17 lang=cpp
 *
 * [17] 电话号码的字母组合
 *
 * https://leetcode.cn/problems/letter-combinations-of-a-phone-number/description/
 *
 * algorithms
 * Medium (58.50%)
 * Likes:    2662
 * Dislikes: 0
 * Total Accepted:    767.6K
 * Total Submissions: 1.3M
 * Testcase Example:  '"23"'
 *
 * 给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
 * 
 * 给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
 * 
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：digits = "23"
 * 输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：digits = ""
 * 输出：[]
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：digits = "2"
 * 输出：["a","b","c"]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 0 <= digits.length <= 4
 * digits[i] 是范围 ['2', '9'] 的一个数字。
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    vector<string> letterCombinations(string digits) {
        if (digits.size() == 0) {
            return {};
        }
        std::vector<std::string> ans;

        this->search(0, digits, "", ans);

        return ans;
    }

private:
    std::unordered_map<char, std::string> m_map {
        {'2', "abc"},
        {'3', "def"},
        {'4', "ghi"},
        {'5', "jkl"},
        {'6', "mno"},
        {'7', "pqrs"},
        {'8', "tuv"},
        {'9', "wxyz"}   
    };

    void search(int idx, std::string& digits, std::string s, 
        std::vector<std::string>& ans) {
        // 终止条件
        if (idx == digits.size()) {
            ans.push_back(s);
            return;
        }

        // 处理
        std::string letters = m_map.at(digits[idx]);
        for (int i = 0; i < letters.size(); i++) {
            // 递归
            this->search(idx + 1, digits, s + letters[i], ans);
        }
    }
};
// @lc code=end

