/*
 * @lc app=leetcode.cn id=547 lang=cpp
 *
 * [547] 省份数量
 *
 * https://leetcode.cn/problems/number-of-provinces/description/
 *
 * algorithms
 * Medium (62.15%)
 * Likes:    1057
 * Dislikes: 0
 * Total Accepted:    288.8K
 * Total Submissions: 464.8K
 * Testcase Example:  '[[1,1,0],[1,1,0],[0,0,1]]'
 *
 * 
 * 
 * 有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c 直接相连，那么城市 a 与城市 c
 * 间接相连。
 * 
 * 省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。
 * 
 * 给你一个 n x n 的矩阵 isConnected ，其中 isConnected[i][j] = 1 表示第 i 个城市和第 j 个城市直接相连，而
 * isConnected[i][j] = 0 表示二者不直接相连。
 * 
 * 返回矩阵中 省份 的数量。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：isConnected = [[1,1,0],[1,1,0],[0,0,1]]
 * 输出：2
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：isConnected = [[1,0,0],[0,1,0],[0,0,1]]
 * 输出：3
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 
 * n == isConnected.length
 * n == isConnected[i].length
 * isConnected[i][j] 为 1 或 0
 * isConnected[i][i] == 1
 * isConnected[i][j] == isConnected[j][i]
 * 
 * 
 * 
 * 
 */

// @lc code=start
// 1.DFS
// 遍历所有城市，对于每个城市，如果该城市尚未被访问过，则从该城市开始深度优先搜索，
// 通过矩阵 isConnected得到与该城市直接相连的城市有哪些，这些城市和该城市属于同一个连通分量，
// 然后对这些城市继续深度优先搜索，直到同一个连通分量的所有城市都被访问到，
// 即可得到一个省份。遍历完全部城市以后，即可得到连通分量的总数，即省份的总数。
class Solution1 {
public:
    int findCircleNum(vector<vector<int>>& isConnected) {
        std::vector<int> visited(isConnected.size(), 0);
        int count = 0;
        for (int i = 0; i < isConnected.size(); i++) {
            if (visited[i] == 0) {
                this->dfs(isConnected, visited, i);
                count++;
            }
        }

        return count;
    }
private:

    void dfs(vector<vector<int>>& isConnected, std::vector<int>& visited, int i) {
        for (int j = 0; j < isConnected.size(); j++) {
            if (isConnected[i][j] == 1 && visited[j] == 0) {
                visited[j] = 1;
                this->dfs(isConnected, visited, j);
            }
        }
    }
};

// 2.BFS
// 遍历所有城市，对于每个城市，如果该城市尚未被访问过，则从该城市开始广度优先搜索，
// 通过矩阵 isConnected得到与该城市直接相连的城市有哪些，这些城市和该城市属于同一个连通分量，
// 然后对这些城市继续深度优先搜索，直到同一个连通分量的所有城市都被访问到，
// 即可得到一个省份。遍历完全部城市以后，即可得到连通分量的总数，即省份的总数。
class Solution2 {
public:
    int findCircleNum(vector<vector<int>>& isConnected) {
        vector<int> visited(isConnected.size(), 0);
        int count = 0;
        std::queue<int> Q;
        for (int i = 0; i < isConnected.size(); i++) {
            if (!visited[i]) {
                Q.push(i);
                while (!Q.empty()) {
                    int j = Q.front(); Q.pop();
                    visited[j] = 1;
                    for (int k = 0; k < isConnected.size(); k++) {
                        if (isConnected[j][k] == 1 && !visited[k]) {
                            Q.push(k);
                        }
                    }
                }
                count++;
            }
        }
        return count;
    }
};

// 3.并查集
// 计算连通分量数的另一个方法是使用并查集。初始时，每个城市都属于不同的连通分量。
// 遍历矩阵 isConnected，如果两个城市之间有相连关系，则它们属于同一个连通分量，
// 对它们进行合并。

// 遍历矩阵 isConnected 的全部元素之后，计算连通分量的总数，即为省份的总数。
class Solution {
public:
    int findCircleNum(vector<vector<int>>& isConnected) {
        int n = isConnected.size();
        m_count = n;
        m_parent.resize(n);
        for (int i = 0; i < n; i++) {
            this->m_parent[i] = i;
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (isConnected[i][j] == 1) {
                    this->union_set(i, j);
                }
            }
        }

        return m_count;
    }

    void union_set(int p, int q) {
        int root_p = this->parent_set(p);
        int root_q = this->parent_set(q);
        if (root_p == root_q) {
            return;
        }
        m_parent[root_p] = root_q;
        m_count--;
    }

    int parent_set(int i) {
        int p = i;
        while(p != m_parent[p]) {
            p = m_parent[p];
        }

        return p;
    }
private:
    int m_count;
    std::vector<int> m_parent;
};

// @lc code=end

