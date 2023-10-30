/*
 * @lc app=leetcode.cn id=773 lang=cpp
 *
 * [773] 滑动谜题
 *
 * https://leetcode.cn/problems/sliding-puzzle/description/
 *
 * algorithms
 * Hard (69.95%)
 * Likes:    321
 * Dislikes: 0
 * Total Accepted:    36.4K
 * Total Submissions: 52.1K
 * Testcase Example:  '[[1,2,3],[4,0,5]]'
 *
 * 在一个 2 x 3 的板上（board）有 5 块砖瓦，用数字 1~5 来表示, 以及一块空缺用 0 来表示。一次 移动 定义为选择 0
 * 与一个相邻的数字（上下左右）进行交换.
 * 
 * 最终当板 board 的结果是 [[1,2,3],[4,5,0]] 谜板被解开。
 * 
 * 给出一个谜板的初始状态 board ，返回最少可以通过多少次移动解开谜板，如果不能解开谜板，则返回 -1 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 
 * 
 * 输入：board = [[1,2,3],[4,0,5]]
 * 输出：1
 * 解释：交换 0 和 5 ，1 步完成
 * 
 * 
 * 示例 2:
 * 
 * 
 * 
 * 
 * 输入：board = [[1,2,3],[5,4,0]]
 * 输出：-1
 * 解释：没有办法完成谜板
 * 
 * 
 * 示例 3:
 * 
 * 
 * 
 * 
 * 输入：board = [[4,1,2],[5,0,3]]
 * 输出：5
 * 解释：
 * 最少完成谜板的最少移动次数是 5 ，
 * 一种移动路径:
 * 尚未移动: [[4,1,2],[5,0,3]]
 * 移动 1 次: [[4,1,2],[0,5,3]]
 * 移动 2 次: [[0,1,2],[4,5,3]]
 * 移动 3 次: [[1,0,2],[4,5,3]]
 * 移动 4 次: [[1,2,0],[4,5,3]]
 * 移动 5 次: [[1,2,3],[4,5,0]]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * board.length == 2
 * board[i].length == 3
 * 0 <= board[i][j] <= 5
 * board[i][j] 中每个值都 不同
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    int slidingPuzzle(vector<vector<int>>& board) {
        std::string init_str;
        for (auto& b : board) {
            for (auto& c : b) {
                init_str += char(c + '0');
            }
        }
        std::string end_str = "123450";

        // [string, count]
        std::queue<std::pair<std::string, int>> queue;
        queue.emplace(init_str, 0);
        // 访问过的字符串
        std::unordered_set<std::string> used = {init_str};

        while (!queue.empty()) {
            auto [str, cnt] = queue.front();
            queue.pop();
            used.insert(str);
            if (str == end_str) {
                return cnt;
            }
            // 字符串变为列表，方便操作
            std::vector<int> arr_str;
            int zero_idx = -1;
            for (int i = 0; i < str.size(); i++) {
                char c = str[i];
                if (c == '0') {
                    zero_idx = i;
                }
                arr_str.push_back(int(c - '0'));
            }
            // 开始移动0
            for (auto& move : m_moves[zero_idx]) {
                std::vector<int> new_arr = arr_str;
                // 交换
                int tmp = new_arr[zero_idx];
                new_arr[zero_idx] = new_arr[move];
                new_arr[move] = tmp;

                // 移动完成，变为字符串
                std::string new_s;
                for (auto& c : new_arr) {
                    new_s += char(c + '0');
                }
                
                // 如果在访问字符串中没有，则加入队列
                if (!used.count(new_s)) {
                    queue.emplace(new_s, cnt + 1);
                }

            }
        }

        return -1;
    }
private:
    // 方向向量
    // 如果0位于下标0位置，则可以向下标1和下标3互换位置
    // 如果0位于下标1位置，则可以向下标0、下标2和下标4互换位置
    std::vector<std::vector<int>> m_moves = {{1, 3}, {0, 2, 4}, {1, 5}, {0, 4}, {1, 3, 5}, {2, 4}};

};
// @lc code=end

