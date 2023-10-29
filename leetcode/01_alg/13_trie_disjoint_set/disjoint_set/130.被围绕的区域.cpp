/*
 * @lc app=leetcode.cn id=130 lang=cpp
 *
 * [130] 被围绕的区域
 *
 * https://leetcode.cn/problems/surrounded-regions/description/
 *
 * algorithms
 * Medium (46.28%)
 * Likes:    1045
 * Dislikes: 0
 * Total Accepted:    247.7K
 * Total Submissions: 535.2K
 * Testcase Example:  '[["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]'
 *
 * 给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' ，找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X'
 * 填充。
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：board =
 * [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
 * 输出：[["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
 * 解释：被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上，或不与边界上的 'O' 相连的 'O'
 * 最终都会被填充为 'X'。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：board = [["X"]]
 * 输出：[["X"]]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * m == board.length
 * n == board[i].length
 * 1 
 * board[i][j] 为 'X' 或 'O'
 * 
 * 
 * 
 * 
 */

// @lc code=start
// 并查集
// 把所有边界上的 O 看做一个连通区域。遇到 O 就执行并查集合并操作，这样所有的 O 就会被分成两类
//     1.和边界上的 O 在一个连通区域内的。这些 O 保留。
//     2.不和边界上的 O 在一个连通区域内的。这些 O 就是被包围的，替换。
// 由于并查集一般用一维数组来记录，方便查找 parants，所以将二维坐标用 node 函数转化为一维坐标。
class Solution {
public:
    void solve(vector<vector<char>>& board) {
        if (board.size() == 0) {
            return;
        }

        int m = board.size();
        int n = board[0].size();
        
        //  初始化并查集, 最后增加一个虚拟结点，用于比较
        m_count = 0;
        int dummy_node = m * n;
        m_parent.resize(m * n + 1);
        for (int i = 0; i < m * n + 1; i++) {
            this->m_parent[i] = i;
            m_count++;
        }
        this->m_parent[dummy_node] = dummy_node;

        int dx[] = {-1, 1, 0, 0};
        int dy[] = {0, 0, -1, 1};
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                // 遇到O，进行并查集操作合并
                if (board[i][j] == 'O') {
                    // 边界上的O,把它和dummy_node 合并成一个连通区域.
                    if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
                        this->union_set(i * n + j, dummy_node);
                    } else {
                        // 和上下左右合并成一个连通区域.
                        for (int k = 0; k < 4; k++) {
                            int x = i + dx[k];
                            int y = j + dy[k];
                            if (board[x][y] == 'O') {
                                this->union_set(i * n + j, x * n + y);
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    // 和dummy_node 在一个连通区域的,那么就是O
                    if (this->parent_set(i * n + j) == this->parent_set(dummy_node)) {
                        board[i][j] = 'O';
                    } else {
                        board[i][j] = 'X';
                    }
                }
            }

        }


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
        // 路径压缩
        while (m_parent[i] != i) {
            int x = i;
            i = m_parent[i];
            m_parent[x] = p;
        }

        return p;
    }
private:
    int m_count;
    std::vector<int> m_parent;
};
// @lc code=end

