/*
 * @lc app=leetcode.cn id=121 lang=cpp
 *
 * [121] 买卖股票的最佳时机
 *
 * https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/description/
 *
 * algorithms
 * Easy (57.72%)
 * Likes:    3306
 * Dislikes: 0
 * Total Accepted:    1.3M
 * Total Submissions: 2.2M
 * Testcase Example:  '[7,1,5,3,6,4]'
 *
 * 给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
 * 
 * 你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
 * 
 * 返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：[7,1,5,3,6,4]
 * 输出：5
 * 解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
 * ⁠    注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：prices = [7,6,4,3,1]
 * 输出：0
 * 解释：在这种情况下, 没有交易完成, 所以最大利润为 0。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 
 * 0 
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.暴力搜索，超时
    int maxProfit1(vector<int>& prices) {
        int max_profit = 0;
        int tmp_max = 0;

        for (int i = 0; i < prices.size() - 1; i++) {
            tmp_max = 0;
            for (int j = i + 1; j < prices.size(); j++) {
                if (max_profit < prices[j] - prices[i]) {
                    max_profit = prices[j] - prices[i];
                }
            }
        }

        return max_profit;
    }

    // 2.一次遍历
    // 维护一个当前最小值和最大利润
    int maxProfit(vector<int>& prices) {
        int max_profit = 0;
        int curr_min_prices = prices[0];

        // 开始循环数组
        for (int i = 1; i < prices.size(); i++) {
            // 如果小于当前的数字，则替换
            // 否则，更新最大利润
            if (prices[i] <= curr_min_prices) {
                curr_min_prices = prices[i];
            } else {
                if (max_profit < prices[i] - curr_min_prices) {
                    max_profit = prices[i] - curr_min_prices;
                }
            }
        }

        return max_profit;
    }
};
// @lc code=end

