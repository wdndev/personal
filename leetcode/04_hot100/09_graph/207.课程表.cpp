/*
 * @lc app=leetcode.cn id=207 lang=cpp
 *
 * [207] 课程表
 *
 * https://leetcode.cn/problems/course-schedule/description/
 *
 * algorithms
 * Medium (53.87%)
 * Likes:    1833
 * Dislikes: 0
 * Total Accepted:    356.5K
 * Total Submissions: 662K
 * Testcase Example:  '2\n[[1,0]]'
 *
 * 你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。
 * 
 * 在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi]
 * ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。
 * 
 * 
 * 例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。
 * 
 * 
 * 请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：numCourses = 2, prerequisites = [[1,0]]
 * 输出：true
 * 解释：总共有 2 门课程。学习课程 1 之前，你需要完成课程 0 。这是可能的。
 * 
 * 示例 2：
 * 
 * 
 * 输入：numCourses = 2, prerequisites = [[1,0],[0,1]]
 * 输出：false
 * 解释：总共有 2 门课程。学习课程 1 之前，你需要先完成​课程 0 ；并且学习课程 0 之前，你还应先完成课程 1 。这是不可能的。
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= numCourses <= 2000
 * 0 <= prerequisites.length <= 5000
 * prerequisites[i].length == 2
 * 0 <= ai, bi < numCourses
 * prerequisites[i] 中的所有课程对 互不相同
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.广度优先遍历，拓扑排序
    bool canFinish1(int numCourses, vector<vector<int>>& prerequisites) {
        // 入度表
        std::vector<int> indegrees(numCourses, 0);
        // 邻接矩阵
        std::vector<std::vector<int>> adjacency(numCourses);
        // 队列
        std::queue<int> que;
        
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
            numCourses--;

            // 遍历邻接矩阵，将入度为0的结点加入队列
            for (auto curr : adjacency[pre]) {
                indegrees[curr]--;
                if (indegrees[curr] == 0) {
                    que.push(curr);
                }
            }
            
        }

        return numCourses == 0;
    }

    // 2.DPS判环
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        // 邻接表
        std::vector<std::vector<int>> adjacency(numCourses);
        
        // 构造邻接表
        for (auto& vec : prerequisites) {
            adjacency[vec[1]].push_back(vec[0]);
        }

        // 访问状态表
        // 0 : 未被访问过
        // 1 : 正在被当前结点访问
        // 2 ：已经访问过
        std::vector<int> visited(numCourses, 0);

        // dfs遍历
        for (int i = 0; i < numCourses; i++) {
            if (!this->dfs(adjacency, visited, i)) {
                return false;
            }
        }

        return true;
    }
    
    bool dfs(std::vector<std::vector<int>>& adjacency, std::vector<int>& visited, int i) {
        if (visited[i] == 1) {
            return false;
        }

        if (visited[i] == 2) {
            return true;
        }

        visited[i] = 1;
        // 遍历邻接矩阵
        for (auto curr : adjacency[i]) {
            if (!this->dfs(adjacency, visited, curr)) {
                return false;
            }
        }
        visited[i] = 2;
        return true;
    }
};
// @lc code=end

