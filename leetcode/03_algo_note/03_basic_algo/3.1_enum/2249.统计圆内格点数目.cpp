/*
 * @lc app=leetcode.cn id=2249 lang=cpp
 *
 * [2249] 统计圆内格点数目
 *
 * https://leetcode.cn/problems/count-lattice-points-inside-a-circle/description/
 *
 * algorithms
 * Medium (53.76%)
 * Likes:    24
 * Dislikes: 0
 * Total Accepted:    10.5K
 * Total Submissions: 19.5K
 * Testcase Example:  '[[2,2,1]]'
 *
 * 给你一个二维整数数组 circles ，其中 circles[i] = [xi, yi, ri] 表示网格上圆心为 (xi, yi) 且半径为 ri
 * 的第 i 个圆，返回出现在 至少一个 圆内的 格点数目 。
 * 
 * 注意：
 * 
 * 
 * 格点 是指整数坐标对应的点。
 * 圆周上的点 也被视为出现在圆内的点。
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 
 * 
 * 输入：circles = [[2,2,1]]
 * 输出：5
 * 解释：
 * 给定的圆如上图所示。
 * 出现在圆内的格点为 (1, 2)、(2, 1)、(2, 2)、(2, 3) 和 (3, 2)，在图中用绿色标识。
 * 像 (1, 1) 和 (1, 3) 这样用红色标识的点，并未出现在圆内。
 * 因此，出现在至少一个圆内的格点数目是 5 。
 * 
 * 示例 2：
 * 
 * 
 * 
 * 
 * 输入：circles = [[2,2,2],[3,4,1]]
 * 输出：16
 * 解释：
 * 给定的圆如上图所示。
 * 共有 16 个格点出现在至少一个圆内。
 * 其中部分点的坐标是 (0, 2)、(2, 0)、(2, 4)、(3, 2) 和 (4, 4) 。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= circles.length <= 200
 * circles[i].length == 3
 * 1 <= xi, yi <= 100
 * 1 <= ri <= min(xi, yi)
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 遍历坐标系中的所有点，根据元的方程过滤出落在圆上面的点
    int countLatticePoints(vector<vector<int>>& circles) {
        int count = 0;
        for (int i = 0; i <= 200; i++) {
            for (int j = 0; j <= 200; j++) {
                for (auto& c : circles) {
                    int x = c[0];
                    int y = c[1];
                    int r = c[2];

                    // 圆心为(a,b)，半径为r的圆的方程 (x-a)^2 + (y-b)^2 = r^2
                    if ((i - x) * ( i - x) + (j - y) * (j - y) <= r * r) {
                        count++;
                        break;
                    }
                }
            }
        }

        return count;
    }
};
// @lc code=end

