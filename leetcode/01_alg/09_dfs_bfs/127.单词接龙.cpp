/*
 * @lc app=leetcode.cn id=127 lang=cpp
 *
 * [127] 单词接龙
 *
 * https://leetcode.cn/problems/word-ladder/description/
 *
 * algorithms
 * Hard (48.31%)
 * Likes:    1313
 * Dislikes: 0
 * Total Accepted:    196K
 * Total Submissions: 405.7K
 * Testcase Example:  '"hit"\n"cog"\n["hot","dot","dog","lot","log","cog"]'
 *
 * 字典 wordList 中从单词 beginWord 和 endWord 的 转换序列 是一个按下述规格形成的序列 beginWord -> s1 ->
 * s2 -> ... -> sk：
 * 
 * 
 * 每一对相邻的单词只差一个字母。
 * 对于 1 <= i <= k 时，每个 si 都在 wordList 中。注意， beginWord 不需要在 wordList 中。
 * sk == endWord
 * 
 * 
 * 给你两个单词 beginWord 和 endWord 和一个字典 wordList ，返回 从 beginWord 到 endWord 的 最短转换序列
 * 中的 单词数目 。如果不存在这样的转换序列，返回 0 。
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：beginWord = "hit", endWord = "cog", wordList =
 * ["hot","dot","dog","lot","log","cog"]
 * 输出：5
 * 解释：一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog", 返回它的长度 5。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：beginWord = "hit", endWord = "cog", wordList =
 * ["hot","dot","dog","lot","log"]
 * 输出：0
 * 解释：endWord "cog" 不在字典中，所以无法进行转换。
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= beginWord.length <= 10
 * endWord.length == beginWord.length
 * 1 <= wordList.length <= 5000
 * wordList[i].length == beginWord.length
 * beginWord、endWord 和 wordList[i] 由小写英文字母组成
 * beginWord != endWord
 * wordList 中的所有字符串 互不相同
 * 
 * 
 */

// @lc code=start
// 最短转换序列 -> 广度优先搜索 -> 图搜索
class Solution {
public:
    // 1. BFS + 建图
    // 映射： 先给每一个单词标号，即给每个单词分配一个 id。创建一个由单词 word 到 id 对应的映射 wordId，
    //       并将 beginWord 与 wordList 中所有的单词都加入这个映射中。
    //       之后检查 endWord 是否在该映射内，若不存在，则输入无解。可以使用哈希表实现上面的映射关系。
    // 建图：可以创建虚拟节点。对于单词 hit，创建三个虚拟节点 *it、h*t、hi*，
    //       并让 hit 向这三个虚拟节点分别连一条边即可。如果一个单词能够转化为 hit，那么该单词必然会连接到这三个虚拟节点之一。
    //       对于每一个单词，我们枚举它连接到的虚拟节点，把该单词对应的 id 与这些虚拟节点对应的 id 相连即可。
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        for (std::string& word : wordList) {
            this->add_edge(word);
        }
        this->add_edge(beginWord);

        if (!m_word_id.count(endWord)) {
            return 0;
        }

        std::vector<int> dis(m_node_num, INT_MAX);
        int begin_id = m_word_id[beginWord];
        int end_id = m_word_id[endWord];

        std::queue<int> que;
        que.push(begin_id);
        while (!que.empty()) {
            int x = que.front();
            que.pop();

            if (x == end_id) {
                return dis[end_id] / 2 + 1;
            }

            for (int& it : m_edge[x]) {
                if (dis[it] == INT_MAX) {
                    dis[it] = dis[x] + 1;
                    que.push(it);
                }
            }
        }

        return 0;
    }
private:
    std::unordered_map<std::string, int> m_word_id;
    std::vector<std::vector<int>> m_edge;
    int m_node_num;

    void add_word(std::string& word) {
        if (!m_word_id.count(word)) {
            m_word_id[word] = m_node_num++;
            m_edge.emplace_back();
        }
    }

    void add_edge(std::string& word) {
        this->add_word(word);
        int id1 = m_word_id[word];

        for (char& ch : word) {
            char tmp = ch;
            ch = '*';
            this->add_word(word);

            int id2 = m_word_id[word];
            m_edge[id1].push_back(id2);
            m_edge[id2].push_back(id1);
            ch = tmp;
        }
    }
};
// @lc code=end

