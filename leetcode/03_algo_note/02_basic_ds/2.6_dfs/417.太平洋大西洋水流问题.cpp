/*
 * @lc app=leetcode.cn id=417 lang=cpp
 *
 * [417] 太平洋大西洋水流问题
 *
 * https://leetcode.cn/problems/pacific-atlantic-water-flow/description/
 *
 * algorithms
 * Medium (56.28%)
 * Likes:    660
 * Dislikes: 0
 * Total Accepted:    92.1K
 * Total Submissions: 163.6K
 * Testcase Example:  '[[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]'
 *
 * 有一个 m × n 的矩形岛屿，与 太平洋 和 大西洋 相邻。 “太平洋” 处于大陆的左边界和上边界，而 “大西洋” 处于大陆的右边界和下边界。
 * 
 * 这个岛被分割成一个由若干方形单元格组成的网格。给定一个 m x n 的整数矩阵 heights ， heights[r][c] 表示坐标 (r, c)
 * 上单元格 高于海平面的高度 。
 * 
 * 岛上雨水较多，如果相邻单元格的高度 小于或等于 当前单元格的高度，雨水可以直接向北、南、东、西流向相邻单元格。水可以从海洋附近的任何单元格流入海洋。
 * 
 * 返回网格坐标 result 的 2D 列表 ，其中 result[i] = [ri, ci] 表示雨水从单元格 (ri, ci) 流动
 * 既可流向太平洋也可流向大西洋 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 
 * 
 * 输入: heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
 * 输出: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入: heights = [[2,1],[1,2]]
 * 输出: [[0,0],[0,1],[1,0],[1,1]]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * m == heights.length
 * n == heights[r].length
 * 1 <= m, n <= 200
 * 0 <= heights[r][c] <= 10^5
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
        this->m_heights = heights;
        int rows = heights.size();
        int cols = heights[0].size();
        // 太平洋能到达的位置
        std::vector<std::vector<bool>> pacific(rows, std::vector<bool>(cols, false));
        // 大西洋能到达的位置
        std::vector<std::vector<bool>> altlantic(rows, std::vector<bool>(cols, false));

        for (int i = 0; i < rows; i++) {
            this->dfs(i, 0, pacific);
            this->dfs(i, cols - 1, altlantic);
        }
        for (int j = 0; j < cols; j++) {
            this->dfs(0, j, pacific);
            this->dfs(rows - 1, j, altlantic);
        }
        
        std::vector<std::vector<int>> ans;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (pacific[i][j] && altlantic[i][j]) {
                    ans.push_back({i, j});
                }
            }
        }

        return ans;
    }
private:
    // 方向
    int m_dx[4] = {-1, 1, 0, 0};
    int m_dy[4] = {0, 0, -1, 1};
    std::vector<std::vector<int>> m_heights;
    void dfs(int i, int j, std::vector<std::vector<bool>>& visited) {
        if (visited[i][j]) {
            return;
        }
        visited[i][j] = true;
        for (int k = 0; k < 4; k++) {
            int x = i + m_dx[k];
            int y = j + m_dy[k];

            if (x >=0 && x < visited.size() && y >= 0 && y < visited[i].size()
                &&  m_heights[x][y] >= m_heights[i][j]) {
                this->dfs(x, y, visited);
            }
        }
    }
};
// @lc code=end

