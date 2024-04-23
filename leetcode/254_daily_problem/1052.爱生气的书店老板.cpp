/*
 * @lc app=leetcode.cn id=1052 lang=cpp
 *
 * [1052] 爱生气的书店老板
 *
 * https://leetcode.cn/problems/grumpy-bookstore-owner/description/
 *
 * algorithms
 * Medium (58.18%)
 * Likes:    316
 * Dislikes: 0
 * Total Accepted:    84.5K
 * Total Submissions: 141.2K
 * Testcase Example:  '[1,0,1,2,1,1,7,5]\n[0,1,0,1,0,1,0,1]\n3'
 *
 * 有一个书店老板，他的书店开了 n 分钟。每分钟都有一些顾客进入这家商店。给定一个长度为 n 的整数数组 customers ，其中
 * customers[i] 是在第 i 分钟开始时进入商店的顾客数量，所有这些顾客在第 i 分钟结束后离开。
 * 
 * 在某些时候，书店老板会生气。 如果书店老板在第 i 分钟生气，那么 grumpy[i] = 1，否则 grumpy[i] = 0。
 * 
 * 当书店老板生气时，那一分钟的顾客就会不满意，若老板不生气则顾客是满意的。
 * 
 * 书店老板知道一个秘密技巧，能抑制自己的情绪，可以让自己连续 minutes 分钟不生气，但却只能使用一次。
 * 
 * 请你返回 这一天营业下来，最多有多少客户能够感到满意 。
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：customers = [1,0,1,2,1,1,7,5], grumpy = [0,1,0,1,0,1,0,1], minutes = 3
 * 输出：16
 * 解释：书店老板在最后 3 分钟保持冷静。
 * 感到满意的最大客户数量 = 1 + 1 + 1 + 1 + 7 + 5 = 16.
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：customers = [1], grumpy = [0], minutes = 1
 * 输出：1
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * n == customers.length == grumpy.length
 * 1 <= minutes <= n <= 2 * 10^4
 * 0 <= customers[i] <= 1000
 * grumpy[i] == 0 or 1
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 定长滑动窗口
    // 可以拆成两个问题：
    // 1. 老板不生气时顾客数量之和 s_0.这些顾客可以感到满意
    // 2.长度为minutes的连续数组中，老板生气时的顾客数量之前s_1的最大值max_s1
    // 最终答案为 s_0 + max s_1
    int maxSatisfied(vector<int>& customers, vector<int>& grumpy, int minutes) {
        int s_0 = 0;    // 老板不生气时，顾客满意的人数
        int s_1 = 0;    // 在minutes的连续时间中，老板生气时，顾客数量之和
        int max_s1 = 0;
        
        for (int i = 0; i < customers.size(); i++) {
            // 根据 grumpy 中的值划分
            if (grumpy[i] == 0) {
                s_0 += customers[i];
            } else {
                s_1 += customers[i];
            }
            // 窗口不足 minutes
            if (i < minutes - 1) {
                continue;
            }
            max_s1 = std::max(max_s1, s_1);
            // 窗口最左边元素离开窗口
            s_1 -= grumpy[i - minutes + 1] ? customers[i - minutes + 1] : 0;
        }

        return s_0 + max_s1;
    }
};
// @lc code=end

