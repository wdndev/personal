/*
 * @lc app=leetcode.cn id=1483 lang=cpp
 *
 * [1483] 树节点的第 K 个祖先
 *
 * https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/description/
 *
 * algorithms
 * Hard (46.25%)
 * Likes:    242
 * Dislikes: 0
 * Total Accepted:    21K
 * Total Submissions: 44K
 * Testcase Example:  '["TreeAncestor","getKthAncestor","getKthAncestor","getKthAncestor"]\n' +
  '[[7,[-1,0,0,1,1,2,2]],[3,1],[5,2],[6,3]]'
 *
 * 给你一棵树，树上有 n 个节点，按从 0 到 n-1 编号。树以父节点数组的形式给出，其中 parent[i] 是节点 i 的父节点。树的根节点是编号为
 * 0 的节点。
 * 
 * 树节点的第 k 个祖先节点是从该节点到根节点路径上的第 k 个节点。
 * 
 * 实现 TreeAncestor 类：
 * 
 * 
 * TreeAncestor（int n， int[] parent） 对树和父数组中的节点数初始化对象。
 * getKthAncestor(int node, int k) 返回节点 node 的第 k 个祖先节点。如果不存在这样的祖先节点，返回 -1
 * 。
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 
 * 
 * 输入：
 * ["TreeAncestor","getKthAncestor","getKthAncestor","getKthAncestor"]
 * [[7,[-1,0,0,1,1,2,2]],[3,1],[5,2],[6,3]]
 * 
 * 输出：
 * [null,1,0,-1]
 * 
 * 解释：
 * TreeAncestor treeAncestor = new TreeAncestor(7, [-1, 0, 0, 1, 1, 2, 2]);
 * 
 * treeAncestor.getKthAncestor(3, 1);  // 返回 1 ，它是 3 的父节点
 * treeAncestor.getKthAncestor(5, 2);  // 返回 0 ，它是 5 的祖父节点
 * treeAncestor.getKthAncestor(6, 3);  // 返回 -1 因为不存在满足要求的祖先节点
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= k <= n <= 5 * 10^4
 * parent[0] == -1 表示编号为 0 的节点是根节点。
 * 对于所有的 0 < i < n ，0 <= parent[i] < n 总成立
 * 0 <= node < n
 * 至多查询 5 * 10^4 次
 * 
 * 
 */

// @lc code=start
class TreeAncestor {
public: 
    // 树上倍增算法，最近公共祖先
    // 一步一步往上跳太慢了，先预处理一些节点，快速往上跳，
    // 预处理出每个节点的第 2^i 个祖先节点，即第1,2,4,8,...。由于任意k可以分解成不同2的幂，如13=8+4=1.
    // 算法：
    // 在构造函数中预处理每个节点x的第2^i个祖先节点，记作pa[x][i]，若第2^i个祖先节点不存在，则pa[x][i]=-1.
    // 1.先枚举i，再枚举x。相当于先算出所有爷爷节点，再算法所有爷爷节点的爷爷节点
    // 2.pa[x][0] = parent[x]，即父节点；pa[x][1]=pa[pa[x][0]][0]，即爷爷节点；
    // 3.依次类推，pa[x][i+1]=pa[pa[x][i]][i]，表示x的第2^i个祖先节点的第2^i个祖先节点就是x的第2^{i+1}个祖先节点。
    //   特别的，如果pa[x][i]=-1,则pa[x][i+1]=-1;
    // 4.这里i+1最多为log_2(n).
    // 对于需要找到k的二进制表示中的所有1.可以从小到大枚举i，如果k右移i位后的最低为1，就说明k的二进制从低到高第i位是1，
    // 那么往上跳 2^i 步，将node更新为 pa[node[i]].如果node=-1，则说明第k个祖先节点不存在
    TreeAncestor(int n, vector<int>& parent) {
        int m = 32 - __builtin_clz(n); // n 的二进制长度
        m_pa.resize(n, std::vector<int>(m, -1));
        for (int i = 0; i < n; i++) {
            m_pa[i][0] = parent[i];
        }
        for (int i = 0; i < m - 1; i++) {
            for (int x = 0; x < n; x++) {
                if (int p = m_pa[x][i]; p != -1) {
                    m_pa[x][i + 1] = m_pa[p][i];
                }
            }
        }
    }
    
    int getKthAncestor(int node, int k) {
        int m = 32 - __builtin_clz(k); // k 的二进制长度
        for (int i = 0; i < m; i++) {
            if ((k >> i) & 1) { // k 的二进制从低到高第 i 位是 1
                node = m_pa[node][i];
                if (node < 0) break;
            }
        }
        return node;
    }
private:
    std::vector<std::vector<int>> m_pa;
};

/**
 * Your TreeAncestor object will be instantiated and called as such:
 * TreeAncestor* obj = new TreeAncestor(n, parent);
 * int param_1 = obj->getKthAncestor(node,k);
 */
// @lc code=end

