// @before-stub-for-debug-begin
#include <vector>
#include <string>
#include "commoncppproblem212.h"

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=212 lang=cpp
 *
 * [212] 单词搜索 II
 *
 * https://leetcode.cn/problems/word-search-ii/description/
 *
 * algorithms
 * Hard (43.57%)
 * Likes:    809
 * Dislikes: 0
 * Total Accepted:    96.6K
 * Total Submissions: 221.9K
 * Testcase Example:  '[["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]\n' +
  '["oath","pea","eat","rain"]'
 *
 * 给定一个 m x n 二维字符网格 board 和一个单词（字符串）列表 words， 返回所有二维网格上的单词 。
 * 
 * 单词必须按照字母顺序，通过 相邻的单元格
 * 内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母在一个单词中不允许被重复使用。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：board =
 * [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]],
 * words = ["oath","pea","eat","rain"]
 * 输出：["eat","oath"]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：board = [["a","b"],["c","d"]], words = ["abcb"]
 * 输出：[]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * m == board.length
 * n == board[i].length
 * 1 <= m, n <= 12
 * board[i][j] 是一个小写英文字母
 * 1 <= words.length <= 3 * 10^4
 * 1 <= words[i].length <= 10
 * words[i] 由小写英文字母组成
 * words 中的所有字符串互不相同
 * 
 * 
 */

// @lc code=start

class Trie {
public:
    Trie() : m_children(26), m_is_end(false) {

    }
    
    // 插入字符串，从根开始，判断下面两种情况
    // 1.子节点存在。沿着指针移动到子节点，继续处理下一个字符。
    // 2.子节点不存在。创建一个新的子节点，记录在 children数组的对应位置上，
    //   然后沿着指针移动到子节点，继续搜索下一个字符。
    void insert(string word) {
        Trie* node = this;
        for (auto ch : word) {
            ch -= 'a';
            if (node->m_children[ch] == nullptr) {
                node->m_children[ch] = new Trie();
            }
            node = node->m_children[ch];
        }
        node->m_is_end = true;
    }
    
    // 搜索字符串是否存在
    // 若搜索到了前缀的末尾，就说明字典树中存在该前缀。
    // 此外，若前缀末尾对应节点的 isEnd为真，则说明字典树中存在该字符串。
    bool search(string word) {
        Trie* node = this->search_prefix(word);
        return node != nullptr && node->m_is_end;
    }
    
    bool startsWith(string prefix) {
        Trie* node = this->search_prefix(prefix);
        return node != nullptr;
    }
    
    // 指向子节点的指针数组 children。数组长度为 26，即小写英文字母的数量
    std::vector<Trie*> m_children;
    // 表示该节点是否为字符串的结尾
    bool m_is_end;

    // 查找前缀，对于当前字符对应的子节点，有两种情况：
    // 1.子节点存在。沿着指针移动到子节点，继续搜索下一个字符。
    // 2.子节点不存在。说明字典树中不包含该前缀，返回空指针。
    Trie* search_prefix(std::string& prefix) {
        Trie* node = this;
        for (auto ch : prefix) {
            ch -= 'a';
            if (node->m_children[ch] == nullptr) {
                return nullptr;
            }
            node = node->m_children[ch];
        }
        return node;
    }
};

class Solution {
public:
    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        if (board.size() == 0 || board[0].size() == 0 || words.size() == 0) {
            return {};
        }

        Trie* trie = new Trie();
        for (auto word : words) {
            trie->insert(word);
        }

        int m = board.size();
        int n = board[0].size();
        // m_visited.resize(m);//r行
        // for (int k = 0; k < m; ++k){
        //     m_visited[k].resize(n);//每行为c列
        // }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (trie->m_children[board[i][j] - 'a'] != nullptr) {
                    this->dfs(board, i, j, "", trie->m_children[board[i][j] - 'a']);
                }
            }
        }

        vector<string> ans;
        for (auto & word: m_result) {
            ans.emplace_back(word);
        }
        return ans;  
    }
private:
    std::vector<std::vector<int>> m_visited;
    std::set<std::string> m_result;

    void dfs(vector<vector<char>>& board, int i, int j, std::string curr_word, Trie* trie) {
        std::string curr_str = curr_word + board[i][j];
        if (trie->m_is_end) {
            m_result.insert(curr_str);
        }

        // m_visited[i][j] = 1;
        std::vector<int> dx = {-1, 1, 0, 0};
        std::vector<int> dy = {0, 0, -1, 1};

        char ch = board[i][j];
        board[i][j] = '@';
        for (int k = 0; k < 4; k++) {
            int x = i + dx[k];
            int y = j + dy[k];
            if (x >= 0 && x < board.size() && y >= 0 && y < board[0].size() 
                && board[x][y] != '@' && trie->m_children[board[x][y] - 'a'] != nullptr) {
                this->dfs(board, x, y, curr_str, trie->m_children[board[x][y] - 'a']);
            }
        }

        // m_visited[i][j] = 0;
        board[i][j] = ch;
    }
};
// @lc code=end

