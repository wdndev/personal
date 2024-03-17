/*
 * @lc app=leetcode.cn id=310 lang=cpp
 *
 * [310] 最小高度树
 *
 * https://leetcode.cn/problems/minimum-height-trees/description/
 *
 * algorithms
 * Medium (42.60%)
 * Likes:    894
 * Dislikes: 0
 * Total Accepted:    75.1K
 * Total Submissions: 172.9K
 * Testcase Example:  '4\n[[1,0],[1,2],[1,3]]'
 *
 * 树是一个无向图，其中任何两个顶点只通过一条路径连接。 换句话说，一个任何没有简单环路的连通图都是一棵树。
 * 
 * 给你一棵包含 n 个节点的树，标记为 0 到 n - 1 。给定数字 n 和一个有 n - 1 条无向边的 edges
 * 列表（每一个边都是一对标签），其中 edges[i] = [ai, bi] 表示树中节点 ai 和 bi 之间存在一条无向边。
 * 
 * 可选择树中任何一个节点作为根。当选择节点 x 作为根节点时，设结果树的高度为 h 。在所有可能的树中，具有最小高度的树（即，min(h)）被称为
 * 最小高度树 。
 * 
 * 请你找到所有的 最小高度树 并按 任意顺序 返回它们的根节点标签列表。
 * 树的 高度 是指根节点和叶子节点之间最长向下路径上边的数量。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：n = 4, edges = [[1,0],[1,2],[1,3]]
 * 输出：[1]
 * 解释：如图所示，当根是标签为 1 的节点时，树的高度是 1 ，这是唯一的最小高度树。
 * 
 * 示例 2：
 * 
 * 
 * 输入：n = 6, edges = [[3,0],[3,1],[3,2],[3,4],[5,4]]
 * 输出：[3,4]
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= n <= 2 * 10^4
 * edges.length == n - 1
 * 0 <= ai, bi < n
 * ai != bi
 * 所有 (ai, bi) 互不相同
 * 给定的输入 保证 是一棵树，并且 不会有重复的边
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // BFS
    // 1. 首先，我们看了样例，发现这个树并不是二叉树，是多叉树。
    // 2. 然后，可能想到的解法是：根据题目的意思，就挨个节点遍历bfs，统计下每个节点的高度，
    //     然后用map存储起来，后面查询这个高度的集合里最小的就可以了。
    // 3. 但是这样会超时的。
    // 4. 于是我们看图（题目介绍里面的图）分析一下，发现，越是靠里面的节点越有可能是最小高度树。
    // 5. 所以，我们可以这样想，我们可以倒着来。
    // 6. 我们从边缘开始，先找到所有出度为1的节点，然后把所有出度为1的节点进队列，然后不断地bfs，
    //    最后找到的就是两边同时向中间靠近的节点，那么这个中间节点就相当于把整个距离二分了，
    //    那么它当然就是到两边距离最小的点啦，也就是到其他叶子节点最近的节点了。
    // 7. 然后，就可以写代码了。
    vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
        std::vector<int> ans;
        // 如果只有一个结点，那么他就是最小高度树
        if (n == 1) {
            ans.push_back(0);
            return ans;
        }
        // 各个结点的出度表
        std::vector<int> degree(n, 0);
        // 建立图关系，在每个结点种存储相连结点
        std::vector<std::vector<int>> map(n);
        for (auto& edge : edges) {
            degree[edge[0]]++;
            degree[edge[1]]++;

            // 添加相邻结点
            map[edge[0]].push_back(edge[1]);
            map[edge[1]].push_back(edge[0]);
        }

        // 建立队列
        std:queue<int> queue;
        // 把所有出度为1的结点，也就是叶子结点添加到队列种
        for (int i = 0; i < n; i++) {
            if (degree[i] == 1) {
                queue.push(i);
            }
        }

        while (!queue.empty()) {
            // 注意，每层都要清除一下，这样最后保存的就是最终的最小高度树
            ans.clear();
            // 每一层的节点数
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int cur = queue.front();
                queue.pop();

                // 把当前节点加入结果集，不要有疑问，为什么当前只是叶子节点为什么要加入结果集呢?
                // 因为我们每次循环都会新建一个list，所以最后保存的就是最后一个状态下的叶子节点，
                // 这也是很多题解里面所说的剪掉叶子节点的部分，你可以想象一下图，每层遍历完，
                // 都会把该层（也就是叶子节点层）这一层从队列中移除掉，
                // 不就相当于把原来的图给剪掉一圈叶子节点，形成一个缩小的新的图吗
                ans.push_back(cur);

                // 经典的BFS
                // 把当前节点的相邻接点都拿出来，把它们的出度都减1，因为当前节点已经不存在了，
                // 所以，它的相邻节点们就有可能变成叶子节点
                std::vector<int> neighbors = map[cur];
                for (int neighbor : neighbors) {
                    degree[neighbor]--;
                    if (degree[neighbor] == 1) {
                        queue.push(neighbor);
                    }
                }
            }
        }

        return ans;
    }
};
// @lc code=end

