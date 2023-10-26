/*
 * @lc app=leetcode.cn id=22 lang=cpp
 *
 * [22] 括号生成
 *
 * https://leetcode.cn/problems/generate-parentheses/description/
 *
 * algorithms
 * Medium (77.49%)
 * Likes:    3410
 * Dislikes: 0
 * Total Accepted:    753.8K
 * Total Submissions: 972.8K
 * Testcase Example:  '3'
 *
 * 数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：n = 3
 * 输出：["((()))","(()())","(())()","()(())","()()()"]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：n = 1
 * 输出：["()"]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= n <= 8
 * 
 * 
 */

// @lc code=start
// 1.DFS
// 2.BFS
class SolutionDFS {
public:
    // DFS
    vector<string> generateParenthesis(int n) {
        std::vector<std::string> ans;
        if (n == 0) {
            return ans;
        }

        this->dfs("", n, n, ans);

        return ans;
    }

private:
    void dfs(std::string curr_str, int left, int right, std::vector<std::string>& ans) {
        // 终止条件，
        if (left == 0 && right == 0) {
            ans.emplace_back(curr_str);
            return;
        }

        // 注意剪枝，（左括号个数严格大于右括号个数）
        if (left > right) {
            return;
        }

        if (left > 0) {
            this->dfs(curr_str + "(", left - 1, right, ans);
        }

        if (right > 0) {
            this->dfs(curr_str + "(", left, right - 1, ans);
        }
    }
};

// BFS实现，需要手动构造结点类
class Node {
public:
    Node (std::string str, int left, int right) {
        m_str = str;
        m_left = left;
        m_right = right;
    }

    // 当前得到的字符串
    std::string m_str;
    // 左括号剩余个数
    int m_left;
    // 右括号剩余个数
    int m_right;
};

class Solution {
public:
    // BFS
    vector<string> generateParenthesis(int n) {
        std::vector<std::string> ans;
        if (n == 0) {
            return ans;
        }

        std::queue<Node> queue;
        queue.push(Node("", n, n));

        while(!queue.empty()) {
            Node curr_node = queue.front();
            queue.pop();

            if (curr_node.m_left == 0 && curr_node.m_right == 0) {
                ans.emplace_back(curr_node.m_str);
            }

            if (curr_node.m_left > 0) {
                queue.push(Node(curr_node.m_str + "(", curr_node.m_left - 1, curr_node.m_right));
            }

            if (curr_node.m_right > 0 && curr_node.m_left < curr_node.m_right) {
                queue.push(Node(curr_node.m_str + ")", curr_node.m_left, curr_node.m_right - 1));
            }
        }

        return ans;
    }

};
// @lc code=end

