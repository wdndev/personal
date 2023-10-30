// @before-stub-for-debug-begin
#include <vector>
#include <string>
#include "commoncppproblem37.h"

using namespace std;
// @before-stub-for-debug-end

/*
 * @lc app=leetcode.cn id=37 lang=cpp
 *
 * [37] 解数独
 *
 * https://leetcode.cn/problems/sudoku-solver/description/
 *
 * algorithms
 * Hard (67.59%)
 * Likes:    1754
 * Dislikes: 0
 * Total Accepted:    225.1K
 * Total Submissions: 333K
 * Testcase Example:  '[["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]'
 *
 * 编写一个程序，通过填充空格来解决数独问题。
 * 
 * 数独的解法需 遵循如下规则：
 * 
 * 
 * 数字 1-9 在每一行只能出现一次。
 * 数字 1-9 在每一列只能出现一次。
 * 数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。（请参考示例图）
 * 
 * 
 * 数独部分空格内已填入了数字，空白格用 '.' 表示。
 * 
 * 
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：board =
 * [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
 * 
 * 输出：[["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
 * 解释：输入的数独如上图所示，唯一有效的解决方案如下所示：
 * 
 * 
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * board.length == 9
 * board[i].length == 9
 * board[i][j] 是一位数字或者 '.'
 * 题目数据 保证 输入数独仅有一个解
 * 
 * 
 * 
 * 
 * 
 */

// @lc code=start
// DFS 回溯解法
class Solution {
public:
    void solveSudoku(vector<vector<char>>& board) {
        if (board.size() == 0) {
            return;
        }

        this->dfs(board);
    }
private:
    

    bool dfs(vector<vector<char>>& board) {
        for (int i = 0; i < board.size(); i++) {
            for (int j = 0; j < board[0].size(); j++) {
                if (board[i][j] == '.') {
                    // 尝试放入 1~9
                    for (char c = '1'; c <= '9'; c++) {
                        // i, j位置放入c
                        board[i][j] = c;
                        // 判断数独是否有效
                        if (this->isValidSudoku(board) && this->dfs(board))
                            return true;
                        // 回溯
                        board[i][j] = '.';
                    }

                    return false;
                }
            }
        }

        return true;
    }


    bool isValidSudoku(vector<vector<char>>& board) {
        // 使用哈希表记录每一行、每一列和每一个小九宫格中，每个数字出现的次数
        // 只需要遍历数独一次，在遍历的过程中更新哈希表中的计数，并判断是否满足有效的数独条件即可
        
        // 每一行的hash表
        int rows[9][9];
        // 每一列的hash表
        int columns[9][9];
        // 每一个小方格的hash表
        int subboxes[3][3][9];
        
        memset(rows, 0, sizeof(rows));
        memset(columns, 0, sizeof(columns));
        memset(subboxes, 0, sizeof(subboxes));
        

        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                char c = board[i][j];
                if (c != '.') {
                    int idx = c - '0' - 1;
                    rows[i][idx]++;
                    columns[j][idx]++;
                    subboxes[i / 3][j / 3][idx]++;

                    if (rows[i][idx] > 1 || columns[j][idx] > 1 || subboxes[i / 3][j / 3][idx] > 1) {
                        return false;
                    }
                }
            }
        }

        return true;
    }
};
// @lc code=end

