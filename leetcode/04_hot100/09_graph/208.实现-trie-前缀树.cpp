/*
 * @lc app=leetcode.cn id=208 lang=cpp
 *
 * [208] 实现 Trie (前缀树)
 *
 * https://leetcode.cn/problems/implement-trie-prefix-tree/description/
 *
 * algorithms
 * Medium (71.92%)
 * Likes:    1542
 * Dislikes: 0
 * Total Accepted:    288.4K
 * Total Submissions: 401.1K
 * Testcase Example:  '["Trie","insert","search","search","startsWith","insert","search"]\n' +
  '[[],["apple"],["apple"],["app"],["app"],["app"],["app"]]'
 *
 * Trie（发音类似 "try"）或者说 前缀树
 * 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。
 * 
 * 请你实现 Trie 类：
 * 
 * 
 * Trie() 初始化前缀树对象。
 * void insert(String word) 向前缀树中插入字符串 word 。
 * boolean search(String word) 如果字符串 word 在前缀树中，返回 true（即，在检索之前已经插入）；否则，返回
 * false 。
 * boolean startsWith(String prefix) 如果之前已经插入的字符串 word 的前缀之一为 prefix ，返回 true
 * ；否则，返回 false 。
 * 
 * 
 * 
 * 
 * 示例：
 * 
 * 
 * 输入
 * ["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
 * [[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
 * 输出
 * [null, null, true, false, true, null, true]
 * 
 * 解释
 * Trie trie = new Trie();
 * trie.insert("apple");
 * trie.search("apple");   // 返回 True
 * trie.search("app");     // 返回 False
 * trie.startsWith("app"); // 返回 True
 * trie.insert("app");
 * trie.search("app");     // 返回 True
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 
 * word 和 prefix 仅由小写英文字母组成
 * insert、search 和 startsWith 调用次数 总计 不超过 3 * 10^4 次
 * 
 * 
 */

// @lc code=start
class Trie {
public:
    Trie() : m_children(26), m_is_end(false) {

    }
    
    // 插入字符串，从根开始，判断下面两种情况：
    // 1.子节点存在。沿着指针移动到子节点，继续处理下一个字符
    // 2.子节点不存在。创建一个新的子节点，记录在children数组的对应位置上，
    //   然后沿着指针移动到子节点，继续搜索下一个字符
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
    // 此外，若前缀末尾对应结点 is_end 为真，则说明字典树中存在该字符串
    bool search(string word) {
        Trie* node = this->search_prefix(word);
        return node != nullptr && node->m_is_end;
    }
    
    // 搜索前缀
    // 如果之前已经插入的字符串 word 的前缀之一为 prefix ，返回 true ；
    // 否则，返回 false 。
    bool startsWith(string prefix) {
        Trie* node = this->search_prefix(prefix);
        return node != nullptr;
    }

private:
    // 指向子节点的指针数组 children，数组长度为26，即小写英文字母的数量
    std::vector<Trie*> m_children;
    // 表示该节点是否为字符串的结尾
    bool m_is_end;

    // 查找前缀，对于当前字符对应的子节点，有两种情况
    // 1.子节点存在。沿着指针移动到子节点，继续搜索下一个字符
    // 2.子节点不存在。说明字典树中不包含该前缀，返回空指针
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

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */
// @lc code=end

