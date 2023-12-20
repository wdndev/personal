/*
 * @lc app=leetcode.cn id=135 lang=cpp
 *
 * [135] 分发糖果
 *
 * https://leetcode.cn/problems/candy/description/
 *
 * algorithms
 * Hard (49.57%)
 * Likes:    1418
 * Dislikes: 0
 * Total Accepted:    269.2K
 * Total Submissions: 543.7K
 * Testcase Example:  '[1,0,2]'
 *
 * n 个孩子站成一排。给你一个整数数组 ratings 表示每个孩子的评分。
 * 
 * 你需要按照以下要求，给这些孩子分发糖果：
 * 
 * 
 * 每个孩子至少分配到 1 个糖果。
 * 相邻两个孩子评分更高的孩子会获得更多的糖果。
 * 
 * 
 * 请你给每个孩子分发糖果，计算并返回需要准备的 最少糖果数目 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：ratings = [1,0,2]
 * 输出：5
 * 解释：你可以分别给第一个、第二个、第三个孩子分发 2、1、2 颗糖果。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：ratings = [1,2,2]
 * 输出：4
 * 解释：你可以分别给第一个、第二个、第三个孩子分发 1、2、1 颗糖果。
 * ⁠    第三个孩子只得到 1 颗糖果，这满足题面中的两个条件。
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * n == ratings.length
 * 1 <= n <= 2 * 10^4
 * 0 <= ratings[i] <= 2 * 10^4
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 两次遍历，分别处理
    // 左规则：当 ratings[i−1]<ratings[i] 时，i 号学生的糖果数量将比i − 1 号孩子的糖果数量多。
    // 右规则：当 ratings[i]>ratings[i+1]时，i 号学生的糖果数量将比 i + 1 号孩子的糖果数量多。
    int candy(vector<int>& ratings) {
        int n = ratings.size();
        // 初始化，人手最少一个
        std::vector<int> sweets(n, 1);
        
        // 第一遍遍历，左规则
        for (int i = 1; i < n; i++) {
            if (ratings[i] > ratings[i - 1]) {
                sweets[i] = sweets[i - 1] + 1;
            }
        }

        // 第二次遍历，右规则
        for (int i = n - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1]) {
                sweets[i] = std::max(sweets[i], sweets[i + 1] + 1);
            }
        }

        // 求和返回
        return std::accumulate(sweets.begin(), sweets.end(), 0);
    }
};
// @lc code=end

