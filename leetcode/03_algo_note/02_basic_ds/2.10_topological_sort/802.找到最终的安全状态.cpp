/*
 * @lc app=leetcode.cn id=802 lang=cpp
 *
 * [802] 找到最终的安全状态
 *
 * https://leetcode.cn/problems/find-eventual-safe-states/description/
 *
 * algorithms
 * Medium (59.12%)
 * Likes:    429
 * Dislikes: 0
 * Total Accepted:    48.1K
 * Total Submissions: 81.3K
 * Testcase Example:  '[[1,2],[2,3],[5],[0],[5],[],[]]'
 *
 * 有一个有 n 个节点的有向图，节点按 0 到 n - 1 编号。图由一个 索引从 0 开始 的 2D 整数数组 graph表示，
 * graph[i]是与节点 i 相邻的节点的整数数组，这意味着从节点 i 到 graph[i]中的每个节点都有一条边。
 * 
 * 如果一个节点没有连出的有向边，则该节点是 终端节点 。如果从该节点开始的所有可能路径都通向 终端节点 ，则该节点为 安全节点 。
 * 
 * 返回一个由图中所有 安全节点 组成的数组作为答案。答案数组中的元素应当按 升序 排列。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 
 * 
 * 输入：graph = [[1,2],[2,3],[5],[0],[5],[],[]]
 * 输出：[2,4,5,6]
 * 解释：示意图如上。
 * 节点 5 和节点 6 是终端节点，因为它们都没有出边。
 * 从节点 2、4、5 和 6 开始的所有路径都指向节点 5 或 6 。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：graph = [[1,2,3,4],[1,2],[3,4],[0,4],[]]
 * 输出：[4]
 * 解释:
 * 只有节点 4 是终端节点，从节点 4 开始的所有路径都通向节点 4 。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * n == graph.length
 * 1 <= n <= 10^4
 * 0 <= graph[i].length <= n
 * 0 <= graph[i][j] <= n - 1
 * graph[i] 按严格递增顺序排列。
 * 图中可能包含自环。
 * 图中边的数目在范围 [1, 4 * 10^4] 内。
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    vector<int> eventualSafeNodes(vector<vector<int>>& graph) {
        // 入度表
        std::vector<int> indegrees(graph.size(), 0);
        // 邻接矩阵
        std::vector<std::vector<int>> adjacency(graph.size());
        // 队列
        std::queue<int> que;

        // 构造邻接矩阵
        for (int i = 0; i < graph.size(); i++) {
            for (auto& v : graph[i]) {
                adjacency[v].push_back(i);
            }
            indegrees[i] = graph[i].size();
        }

        // 将入度为0的结点加入队列
        for (int i = 0; i < graph.size(); i++) {
            if (indegrees[i] == 0) {
                que.push(i);
            }
        }

        // BFS遍历
        while (!que.empty()) {
            int pre = que.front();
            que.pop();
            
            // 遍历邻接矩阵，将入度为0的节点加入队列
            for (auto curr : adjacency[pre]) {
                indegrees[curr]--;
                if (indegrees[curr] == 0) {
                    que.push(curr);
                }
            }
        }

        std::vector<int> ans;
        for (int i = 0; i < graph.size(); i++) {
            if (indegrees[i] == 0) {
                ans.push_back(i);
            }
        }

        return ans;
    }
};
// @lc code=end

