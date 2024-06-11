/*
 * @lc app=leetcode.cn id=419 lang=cpp
 *
 * [419] 甲板上的战舰
 *
 * https://leetcode.cn/problems/battleships-in-a-board/description/
 *
 * algorithms
 * Medium (77.77%)
 * Likes:    290
 * Dislikes: 0
 * Total Accepted:    64.1K
 * Total Submissions: 81.5K
 * Testcase Example:  '[["X",".",".","X"],[".",".",".","X"],[".",".",".","X"]]'
 *
 * 给你一个大小为 m x n 的矩阵 board 表示甲板，其中，每个单元格可以是一艘战舰 'X' 或者是一个空位 '.' ，返回在甲板 board
 * 上放置的 战舰 的数量。
 * 
 * 战舰 只能水平或者垂直放置在 board 上。换句话说，战舰只能按 1 x k（1 行，k 列）或 k x 1（k 行，1 列）的形状建造，其中 k
 * 可以是任意大小。两艘战舰之间至少有一个水平或垂直的空位分隔 （即没有相邻的战舰）。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：board = [["X",".",".","X"],[".",".",".","X"],[".",".",".","X"]]
 * 输出：2
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：board = [["."]]
 * 输出：0
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * m == board.length
 * n == board[i].length
 * 1 <= m, n <= 200
 * board[i][j] 是 '.' 或 'X'
 * 
 * 
 * 
 * 
 * 进阶：你可以实现一次扫描算法，并只使用 O(1) 额外空间，并且不修改 board 的值来解决这个问题吗？
 * 
 */

// @lc code=start
class Solution {
public:
    // 岛屿数量，炸岛屿
    // 使用dfs遍历，遇到陆地，变成海洋
    int countBattleships(vector<vector<char>>& board) {
        int ans = 0;
        for (int i = 0; i < board.size(); i++) {
            for (int j = 0; j < board[0].size(); j++) {
                if (board[i][j] == 'X') {
                    ans++;
                    this->dfs(i, j, board);
                }
            }
        }
        return ans;
    }
private:
    int m_dx[4] = {0, 0, 1, -1};
    int m_dy[4] = {1, -1, 0, 0};
    void dfs(int i, int j, std::vector<std::vector<char>>& board) {
        if (i < 0 || i >= board.size() || j < 0 || j >= board[i].size() || board[i][j] == '.') {
            return;
        }
        board[i][j] = '.';
        for (int k = 0; k < 4; k++) {
            int x = i + m_dx[k];
            int y = j + m_dy[k];
            this->dfs(x, y, board);
        }
    }
};
// @lc code=end

