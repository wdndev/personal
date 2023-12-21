/*
 * @lc app=leetcode.cn id=1710 lang=cpp
 *
 * [1710] 卡车上的最大单元数
 *
 * https://leetcode.cn/problems/maximum-units-on-a-truck/description/
 *
 * algorithms
 * Easy (73.39%)
 * Likes:    131
 * Dislikes: 0
 * Total Accepted:    51.7K
 * Total Submissions: 70.5K
 * Testcase Example:  '[[1,3],[2,2],[3,1]]\n4'
 *
 * 请你将一些箱子装在 一辆卡车 上。给你一个二维数组 boxTypes ，其中 boxTypes[i] = [numberOfBoxesi,
 * numberOfUnitsPerBoxi] ：
 * 
 * 
 * numberOfBoxesi 是类型 i 的箱子的数量。
 * numberOfUnitsPerBoxi 是类型 i 每个箱子可以装载的单元数量。
 * 
 * 
 * 整数 truckSize 表示卡车上可以装载 箱子 的 最大数量 。只要箱子数量不超过 truckSize ，你就可以选择任意箱子装到卡车上。
 * 
 * 返回卡车可以装载 单元 的 最大 总数。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：boxTypes = [[1,3],[2,2],[3,1]], truckSize = 4
 * 输出：8
 * 解释：箱子的情况如下：
 * - 1 个第一类的箱子，里面含 3 个单元。
 * - 2 个第二类的箱子，每个里面含 2 个单元。
 * - 3 个第三类的箱子，每个里面含 1 个单元。
 * 可以选择第一类和第二类的所有箱子，以及第三类的一个箱子。
 * 单元总数 = (1 * 3) + (2 * 2) + (1 * 1) = 8
 * 
 * 示例 2：
 * 
 * 
 * 输入：boxTypes = [[5,10],[2,5],[4,7],[3,9]], truckSize = 10
 * 输出：91
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 
 * 1 i, numberOfUnitsPerBoxi 
 * 1 
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 贪心
    // 按照每个箱子可以装载的单元数量对数组 boxTypes 从大到小排序。
    // 然后优先选取装载单元数量多的箱子。
    int maximumUnits(vector<vector<int>>& boxTypes, int truckSize) {
        if (boxTypes.size() == 0) {
            return 0;
        }

        // 按照每个箱子可以装载的单元数量从大到小排序
        std::sort(boxTypes.begin(), boxTypes.end(), [](const auto& u, const auto& v){
            return u[1] > v[1];
        });

        int res = 0;

        for (const auto& box : boxTypes) {
            // 当前种类的箱子数量少于需求数量，全部加入
            if (box[0] < truckSize) {
                res += box[0] * box[1];
                truckSize -= box[0];
            } else {
                res += truckSize * box[1];
                break;
            }
        }

        return res;
    }
};
// @lc code=end

