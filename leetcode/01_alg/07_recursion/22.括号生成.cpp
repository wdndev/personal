/*
 * @lc app=leetcode.cn id=22 lang=cpp
 *
 * [22] 括号生成
 */

// 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。

// 示例 1：
// 输入：n = 3
// 输出：["((()))","(()())","(())()","()(())","()()()"]

// 示例 2：
// 输入：n = 1
// 输出：["()"]


// @lc code=start
class Solution {
public:
    // 递归
    // 左括号: 随时加，只要不超标
    // 右括号 : 必须之前有左括号，且左括号个数 > 右括号个数
    vector<string> generateParenthesis(int n) {
        result.clear();
        rhis->_generate(0, 0, n, "");
        return result;
    }

    void _generate(int left, int right, int num, std::string s) {
        // 1.terminator
        if (left == num && right == num) {
            result.emplace_back(s);
            // std::cout << s << std::endl;
            return;
        }

        // 2.process current logic

        // 3.drill down
        if (left < num)
            this->_generate(left + 1, right, num, s + "(");
        
        if (left > right)
            this->_generate(left, right + 1, num, s + ")");

        // 4.reverse states
    }
private:
    std::vector<std::string> result;
};
// @lc code=end

