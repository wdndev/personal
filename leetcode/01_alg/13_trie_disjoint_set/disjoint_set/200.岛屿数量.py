"""
 * @lc app=leetcode.cn id=200 lang=cpp
 *
 * [200] 岛屿数量
 *
 * https://leetcode.cn/problems/number-of-islands/description/
 *
 * algorithms
 * Medium (59.68%)
 * Likes:    2336
 * Dislikes: 0
 * Total Accepted:    720.8K
 * Total Submissions: 1.2M
 * Testcase Example:  '[["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]'
 *
 * 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。
 * 
 * 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
 * 
 * 此外，你可以假设该网格的四条边均被水包围。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：grid = [
 * ⁠ ["1","1","1","1","0"],
 * ⁠ ["1","1","0","1","0"],
 * ⁠ ["1","1","0","0","0"],
 * ⁠ ["0","0","0","0","0"]
 * ]
 * 输出：1
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：grid = [
 * ⁠ ["1","1","0","0","0"],
 * ⁠ ["1","1","0","0","0"],
 * ⁠ ["0","0","1","0","0"],
 * ⁠ ["0","0","0","1","1"]
 * ]
 * 输出：3
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * m == grid.length
 * n == grid[i].length
 * 1 
 * grid[i][j] 的值为 '0' 或 '1'
 * 
 * 
 */
"""

# @lc code=start
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        """
        为了求出岛屿的数量，可以扫描整个二维网格。如果一个位置为 1，
        则将其与相邻四个方向上的 111 在并查集中进行合并。

        最终岛屿的数量就是并查集中连通分量的数目。
        """
        if not grid:
            return 0
        
        m = len(grid)
        n = len(grid[0])
        # 初始化并查集
        self.parent = [i for i in range(m * n)]
        self.count =0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    self.parent[i * n + j] = i * n + j
                    self.count += 1

        # 合并
        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    grid[i][j] = "0"
                    for k in range(len(dx)):
                        x, y = i + dx[k], j + dy[k]
                        if 0 <= x < m and 0 <= y < n and grid[x][y] == "1":
                            self._union(i * n + j, x * n + y)

        return self.count

    def _union(self, i, j):
        p1 = self._parent(i)
        p2 = self._parent(j)
        if p1 == p2:
            return
        self.parent[p1] = p2
        self.count -= 1

    def _parent(self, i):
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        # 路径压缩
        while self.parent[i] != i:
            x = i
            i = self.parent[i]
            self.parent[x] = root
        return root
# @lc code=end

