/*
 * @lc app=leetcode.cn id=210 lang=cpp
 *
 * [210] 课程表 II
 *
 * https://leetcode.cn/problems/course-schedule-ii/description/
 *
 * algorithms
 * Medium (57.81%)
 * Likes:    917
 * Dislikes: 0
 * Total Accepted:    216.8K
 * Total Submissions: 374.9K
 * Testcase Example:  '2\n[[1,0]]'
 *
 * 现在你总共有 numCourses 门课需要选，记为 0 到 numCourses - 1。给你一个数组 prerequisites ，其中
 * prerequisites[i] = [ai, bi] ，表示在选修课程 ai 前 必须 先选修 bi 。
 * 
 * 
 * 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示：[0,1] 。
 * 
 * 
 * 返回你为了学完所有课程所安排的学习顺序。可能会有多个正确的顺序，你只要返回 任意一种 就可以了。如果不可能完成所有课程，返回 一个空数组 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：numCourses = 2, prerequisites = [[1,0]]
 * 输出：[0,1]
 * 解释：总共有 2 门课程。要学习课程 1，你需要先完成课程 0。因此，正确的课程顺序为 [0,1] 。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
 * 输出：[0,2,1,3]
 * 解释：总共有 4 门课程。要学习课程 3，你应该先完成课程 1 和课程 2。并且课程 1 和课程 2 都应该排在课程 0 之后。
 * 因此，一个正确的课程顺序是 [0,1,2,3] 。另一个正确的排序是 [0,2,1,3] 。
 * 
 * 示例 3：
 * 
 * 
 * 输入：numCourses = 1, prerequisites = []
 * 输出：[0]
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= numCourses <= 2000
 * 0 <= prerequisites.length <= numCourses * (numCourses - 1)
 * prerequisites[i].length == 2
 * 0 <= ai, bi < numCourses
 * ai != bi
 * 所有[ai, bi] 互不相同
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.广度优先遍历，拓扑排序
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        // 入度表
        std::vector<int> indegrees(numCourses, 0);
        // 邻接矩阵
        std::vector<std::vector<int>> adjacency(numCourses);
        // 队列
        std::queue<int> que;

        std::vector<int> ans;
        
        // 构造邻接矩阵
        for (auto& vec : prerequisites) {
            // 入度++
            indegrees[vec[0]]++;
            adjacency[vec[1]].push_back(vec[0]);
        }

        // 将入度为0的结点加入队列，表示没有任何前置课程
        for (int i = 0; i < numCourses; i++) {
            if (indegrees[i] == 0) {
                que.push(i);
            }
        }

        // BFS遍历，
        while (!que.empty()) {
            int pre = que.front();
            que.pop();
            ans.push_back(pre);

            // 遍历邻接矩阵，将入度为0的结点加入队列
            for (auto curr : adjacency[pre]) {
                indegrees[curr]--;
                if (indegrees[curr] == 0) {
                    que.push(curr);
                }
            }
            
        }

        // 还有顶点未遍历（存在环），无法构成拓扑序列
        if (indegrees.size() != ans.size()) {
            return {};
        } 

        return ans;
    }
};
// @lc code=end

