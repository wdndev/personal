/*
 * @lc app=leetcode.cn id=874 lang=cpp
 *
 * [874] 模拟行走机器人
 *
 * https://leetcode.cn/problems/walking-robot-simulation/description/
 *
 * algorithms
 * Medium (47.28%)
 * Likes:    238
 * Dislikes: 0
 * Total Accepted:    44.3K
 * Total Submissions: 93.7K
 * Testcase Example:  '[4,-1,3]\n[]'
 *
 * 机器人在一个无限大小的 XY 网格平面上行走，从点 (0, 0) 处开始出发，面向北方。该机器人可以接收以下三种类型的命令 commands
 * ：
 * 
 * 
 * -2 ：向左转 90 度
 * -1 ：向右转 90 度
 * 1 <= x <= 9 ：向前移动 x 个单位长度
 * 
 * 
 * 在网格上有一些格子被视为障碍物 obstacles 。第 i 个障碍物位于网格点  obstacles[i] = (xi, yi) 。
 * 
 * 机器人无法走到障碍物上，它将会停留在障碍物的前一个网格方块上，并继续执行下一个命令。
 * 
 * 返回机器人距离原点的 最大欧式距离 的 平方 。（即，如果距离为 5 ，则返回 25 ）
 * 
 * 
 * 注意：
 * 
 * 
 * 北方表示 +Y 方向。
 * 东方表示 +X 方向。
 * 南方表示 -Y 方向。
 * 西方表示 -X 方向。
 * 原点 [0,0] 可能会有障碍物。
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：commands = [4,-1,3], obstacles = []
 * 输出：25
 * 解释：
 * 机器人开始位于 (0, 0)：
 * 1. 向北移动 4 个单位，到达 (0, 4)
 * 2. 右转
 * 3. 向东移动 3 个单位，到达 (3, 4)
 * 距离原点最远的是 (3, 4) ，距离为 3^2 + 4^2 = 25
 * 
 * 示例 2：
 * 
 * 
 * 输入：commands = [4,-1,4,-2,4], obstacles = [[2,4]]
 * 输出：65
 * 解释：机器人开始位于 (0, 0)：
 * 1. 向北移动 4 个单位，到达 (0, 4)
 * 2. 右转
 * 3. 向东移动 1 个单位，然后被位于 (2, 4) 的障碍物阻挡，机器人停在 (1, 4)
 * 4. 左转
 * 5. 向北走 4 个单位，到达 (1, 8)
 * 距离原点最远的是 (1, 8) ，距离为 1^2 + 8^2 = 65
 * 
 * 示例 3：
 * 
 * 
 * 输入：commands = [6,-1,-1,6], obstacles = []
 * 输出：36
 * 解释：机器人开始位于 (0, 0):
 * 1. 向北移动 6 个单位，到达 (0, 6).
 * 2. 右转
 * 3. 右转
 * 4. 向南移动 6 个单位，到达 (0, 0).
 * 机器人距离原点最远的点是 (0, 6)，其距离的平方是 6^2 = 36 个单位。
 * 
 * 提示：
 * 
 * 
 * 1 <= commands.length <= 10^4
 * commands[i] 的值可以取 -2、-1 或者是范围 [1, 9] 内的一个整数。
 * 0 <= obstacles.length <= 10^4
 * -3 * 10^4 <= xi, yi <= 3 * 10^4
 * 答案保证小于 2^31
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    int robotSim(vector<int>& commands, vector<vector<int>>& obstacles) {
        // 方向数组， x后退，y前进，x前进， y后退
        int dirs[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
        // 起始点 (0, 0)
        int px = 0, py = 0;
        // 方向标号d，
        int d = 1;
        // 则往前移动一单位的操作为 x=x+dirs[d][0]，y=y+dirs[i][1]。
        // 向左转的操作为 d=(d+3) mod 4，向右转的操作为 d=(d+1) mod 4。


        std::unordered_set<int> mp;
        for (auto& ob : obstacles) {
            mp.emplace(ob[0] * 60001 + ob[1]);
        }

        int res = 0;
        for (int c : commands) {
            if (c < 0) {    // 转弯
                // 右转d +1， 左转d -1
                d += c == -1 ? 1 : -1;
                d %= 4;
                if (d < 0) {
                    d += 4;
                }
            } else {    // 移动
                for (int i = 0; i < c; i++) {
                    if (mp.count((px + dirs[d][0]) * 60001 + py + dirs[d][1])) {
                        break;
                    }

                    px += dirs[d][0];
                    py += dirs[d][1];
                    res = std::max(res, px * px + py * py);
                }
            }
        }

        return res;
    }
};
// @lc code=end

