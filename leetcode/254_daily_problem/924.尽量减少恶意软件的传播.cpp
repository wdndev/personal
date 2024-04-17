/*
 * @lc app=leetcode.cn id=924 lang=cpp
 *
 * [924] 尽量减少恶意软件的传播
 *
 * https://leetcode.cn/problems/minimize-malware-spread/description/
 *
 * algorithms
 * Hard (36.31%)
 * Likes:    127
 * Dislikes: 0
 * Total Accepted:    21.3K
 * Total Submissions: 51.9K
 * Testcase Example:  '[[1,1,0],[1,1,0],[0,0,1]]\n[0,1]'
 *
 * 给出了一个由 n 个节点组成的网络，用 n × n 个邻接矩阵图 graph 表示。在节点网络中，当 graph[i][j] = 1 时，表示节点 i
 * 能够直接连接到另一个节点 j。 
 * 
 * 一些节点 initial
 * 最初被恶意软件感染。只要两个节点直接连接，且其中至少一个节点受到恶意软件的感染，那么两个节点都将被恶意软件感染。这种恶意软件的传播将继续，直到没有更多的节点可以被这种方式感染。
 * 
 * 假设 M(initial) 是在恶意软件停止传播之后，整个网络中感染恶意软件的最终节点数。
 * 
 * 如果从 initial 中移除某一节点能够最小化 M(initial)， 返回该节点。如果有多个节点满足条件，就返回索引最小的节点。
 * 
 * 请注意，如果某个节点已从受感染节点的列表 initial 中删除，它以后仍有可能因恶意软件传播而受到感染。
 * 
 * 
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：graph = [[1,1,0],[1,1,0],[0,0,1]], initial = [0,1]
 * 输出：0
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：graph = [[1,0,0],[0,1,0],[0,0,1]], initial = [0,2]
 * 输出：0
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：graph = [[1,1,1],[1,1,1],[1,1,1]], initial = [1,2]
 * 输出：1
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 
 * n == graph.length
 * n == graph[i].length
 * 2 <= n <= 300
 * graph[i][j] == 0 或 1.
 * graph[i][j] == graph[j][i]
 * graph[i][i] == 1
 * 1 <= initial.length <= n
 * 0 <= initial[i] <= n - 1
 * initial 中所有整数均不重复
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 要找只包含一个被感染节点的连通块，并且这个连通块最大越好，这样就会使 M(initial)减少最多
    // 1.遍历initial 中的节点x
    // 2.如果x没有被访问过，那么从x开始dfs，用visited数组表示访问过的节点
    // 3.dfs过程中，统计连通块大小size
    // 4.dfs过程中，记录访问到在initial中的节点
    // 5.dfs结束后，如果发现该连通块只有一个在initial中的节点，并且该连通块的大小比最大的连通块更大
    //   那么更新最大连通块的大小，以及答案节点x。如果一样大就更新答案节点的最小值
    // 6.最后，如果没有找到符合要求的节点，返回 min(inital)；否则返回答案节点
    // 用一个哈希表或数组，记录那些点在initial中，从而在dfs中快速判断当前节点是否在initial中
    // 连通块内节点感染：使用状态机记录
    //  - 初始状态为 -1
    //  - 如果状态时 -1，在找到被感染节点x后，状态变为x
    //  - 如果状态时非负数x，在找到另一个被感染的节点后，状态标为-2；如果已经是02，则不变。
    int minMalwareSpread(vector<vector<int>>& graph, vector<int>& initial) {
        std::set<int> state(initial.begin(), initial.end());
        std::vector<int> visited(graph.size());
        int ans = -1;
        int max_size = 0;

        for (int x : initial) {
            if (visited[x]) {
                continue;
            }

            m_node_id = -1;
            m_size = 0;
            this->dfs(x, graph, visited, state);
            if (m_node_id >= 0 && (m_size > max_size || m_size == max_size && m_node_id < ans)) {
                ans = m_node_id;
                max_size = m_size;
            }
        }

        return ans < 0 ? ranges::min(initial) : ans;
    }

    void dfs(int x, vector<vector<int>>& graph, std::vector<int>& visited, std::set<int>& state) {
        visited[x] = true;
        m_size++;
        // 按照状态机更新 node_id
        if (m_node_id != -2 && state.contains(x)) {
            m_node_id = m_node_id == -1 ? x : -2;
        }
        for (int y = 0; y < graph[x].size(); y++) {
            if (graph[x][y] && !visited[y]) {
                this->dfs(y, graph, visited, state);
            }
        }
    }

    int m_node_id;
    int m_size;
};
// @lc code=end

