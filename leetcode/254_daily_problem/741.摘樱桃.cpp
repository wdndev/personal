/*
 * @lc app=leetcode.cn id=741 lang=cpp
 *
 * [741] 摘樱桃
 *
 * https://leetcode.cn/problems/cherry-pickup/description/
 *
 * algorithms
 * Hard (50.89%)
 * Likes:    436
 * Dislikes: 0
 * Total Accepted:    28.3K
 * Total Submissions: 52.8K
 * Testcase Example:  '[[0,1,-1],[1,0,-1],[1,1,1]]'
 *
 * 给你一个 n x n 的网格 grid ，代表一块樱桃地，每个格子由以下三种数字的一种来表示：
 * 
 * 
 * 0 表示这个格子是空的，所以你可以穿过它。
 * 1 表示这个格子里装着一个樱桃，你可以摘到樱桃然后穿过它。
 * -1 表示这个格子里有荆棘，挡着你的路。
 * 
 * 
 * 请你统计并返回：在遵守下列规则的情况下，能摘到的最多樱桃数：
 * 
 * 
 * 从位置 (0, 0) 出发，最后到达 (n - 1, n - 1) ，只能向下或向右走，并且只能穿越有效的格子（即只可以穿过值为 0 或者 1
 * 的格子）；
 * 当到达 (n - 1, n - 1) 后，你要继续走，直到返回到 (0, 0) ，只能向上或向左走，并且只能穿越有效的格子；
 * 当你经过一个格子且这个格子包含一个樱桃时，你将摘到樱桃并且这个格子会变成空的（值变为 0 ）；
 * 如果在 (0, 0) 和 (n - 1, n - 1) 之间不存在一条可经过的路径，则无法摘到任何一个樱桃。
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：grid = [[0,1,-1],[1,0,-1],[1,1,1]]
 * 输出：5
 * 解释：玩家从 (0, 0) 出发：向下、向下、向右、向右移动至 (2, 2) 。
 * 在这一次行程中捡到 4 个樱桃，矩阵变成 [[0,1,-1],[0,0,-1],[0,0,0]] 。
 * 然后，玩家向左、向上、向上、向左返回起点，再捡到 1 个樱桃。
 * 总共捡到 5 个樱桃，这是最大可能值。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：grid = [[1,1,-1],[1,-1,1],[-1,1,1]]
 * 输出：0
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * n == grid.length
 * n == grid[i].length
 * 1 <= n <= 50
 * grid[i][j] 为 -1、0 或 1
 * grid[0][0] != -1
 * grid[n - 1][n - 1] != -1
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // dfs + 记忆化
    // 定义 dfs(t, j, k)表示从 (0, 0) 出发，都走了 t 步，分表到(t-j, j)和(t-k, k)，可以得到的最大樱桃个数
    // A可以往下右走，B也可以，总共有4种情况
    // - dfs(t - 1, j, k) : A 下走，B下
    // - dfs(t - 1, j - 1, k) : A 右， B下
    // - dfs(t - 1, j, k - 1) : A 下，B右
    // - dfs(t - 1, j - 1, k - 1) : A右，B右
    // dfs(t, j, k) = max( ... ) + grid[t-j][j] + grid[t-k][k]
    int cherryPickup(vector<vector<int>>& grid) {
        int n = grid.size();
        vector<vector<vector<int>>> memo(n * 2 - 1, vector<vector<int>>(n, vector<int>(n, -1))); // -1 表示没有计算过

        return max(this->dfs(n * 2 - 2, n - 1, n - 1, memo, grid), 0);
    }

    int dfs(int t, int j, int k, vector<vector<vector<int>>>& memo, vector<vector<int>>& grid) {
        // 不能出界
        if (j < 0 || k < 0 || t < j || t < k || grid[t-j][j] < 0 || grid[t-k][k] < 0) {
            return INT_MIN;
        }
        // 此时 j = k = 0
        if (t == 0) {
            return grid[0][0];
        }
        if (memo[t][j][k] != -1) {
            return memo[t][j][k];
        }

        memo[t][j][k] = max({dfs(t - 1, j, k, memo, grid), 
                             dfs(t - 1, j, k - 1, memo, grid), 
                             dfs(t - 1, j - 1, k, memo, grid), 
                             dfs(t - 1, j - 1, k - 1, memo, grid)}) +
                        grid[t - j][j] + (k != j ? grid[t - k][k] : 0);
        return memo[t][j][k];
    }
};
// @lc code=end

