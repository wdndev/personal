/*
 * @lc app=leetcode.cn id=743 lang=cpp
 *
 * [743] 网络延迟时间
 *
 * https://leetcode.cn/problems/network-delay-time/description/
 *
 * algorithms
 * Medium (56.07%)
 * Likes:    727
 * Dislikes: 0
 * Total Accepted:    124.3K
 * Total Submissions: 221.5K
 * Testcase Example:  '[[2,1,1],[2,3,1],[3,4,1]]\n4\n2'
 *
 * 有 n 个网络节点，标记为 1 到 n。
 * 
 * 给你一个列表 times，表示信号经过 有向 边的传递时间。 times[i] = (ui, vi, wi)，其中 ui 是源节点，vi 是目标节点，
 * wi 是一个信号从源节点传递到目标节点的时间。
 * 
 * 现在，从某个节点 K 发出一个信号。需要多久才能使所有节点都收到信号？如果不能使所有节点收到信号，返回 -1 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 
 * 
 * 输入：times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
 * 输出：2
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：times = [[1,2,1]], n = 2, k = 1
 * 输出：1
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：times = [[1,2,1]], n = 2, k = 2
 * 输出：-1
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= k <= n <= 100
 * 1 <= times.length <= 6000
 * times[i].length == 3
 * 1 <= ui, vi <= n
 * ui != vi
 * 0 <= wi <= 100
 * 所有 (ui, vi) 对都 互不相同（即，不含重复边）
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // Dijkstra 算法
    // 定义g[i][j]表示结点i到结点j这条边的边权。如果没有i到j的边，则g[i][j]=inf
    // 定义 dis[i]表示结点k到结点i的最短路径，一开始dis[k]=0，其余dis[i]=inf表示尚未计算出
    // 目标：计算出最终的 dis 数组
    // 1.更新结点k到其邻居y的最短路径，即更新dis[y] 为 g[k][y]
    // 2.取出除了结点k以外的dis[i]的最小值，假设最小值对应的结点是3.此时可以断言：dis[3]已经是k到3的最短长度了。
    // 3.用结点3到其邻居y的边权g[3][y] 更新 dis[y]: 如果 dis[3]+g[3][y] < dis[y]，那么更新 dis[y]为 dis[3]+g[3][y]，否则不更新
    // 4.取除了结点k，3以外的 dis[i] 的最小值，重复上述过程
    int networkDelayTime(vector<vector<int>>& times, int n, int k) {
        // 邻接矩阵
        std::vector<std::vector<int>> g(n, std::vector<int>(n, INT_MAX / 2));
        for (auto& t : times) {
            g[t[0] - 1][t[1] - 1] = t[2];
        }
        std::vector<int> dis(n, INT_MAX/2);
        std::vector<int> visited(n);
        dis[k - 1] = 0;

        while (true) {
            int x = -1;
            for (int i = 0; i < n; i++) {
                if (!visited[i] && (x < 0 || dis[i] < dis[x])) {
                    x = i;
                }
            }
            if (x < 0) {
                return ranges::max(dis);
            }
            // 结点无法到达
            if (dis[x] == INT_MAX / 2) {
                return -1;
            }
            // 最短路径已经确定
            visited[x] = true;
            // 更新x的邻居y的最短路径
            for (int y = 0; y < n; y++) {
                dis[y] = min(dis[y], dis[x] + g[x][y]);
            }
        }

    }
};
// @lc code=end

