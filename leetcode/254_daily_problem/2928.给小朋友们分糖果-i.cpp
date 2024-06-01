/*
 * @lc app=leetcode.cn id=2928 lang=cpp
 *
 * [2928] 给小朋友们分糖果 I
 *
 * https://leetcode.cn/problems/distribute-candies-among-children-i/description/
 *
 * algorithms
 * Easy (74.24%)
 * Likes:    53
 * Dislikes: 0
 * Total Accepted:    16.1K
 * Total Submissions: 19.7K
 * Testcase Example:  '5\n2'
 *
 * 给你两个正整数 n 和 limit 。
 * 
 * 请你将 n 颗糖果分给 3 位小朋友，确保没有任何小朋友得到超过 limit 颗糖果，请你返回满足此条件下的 总方案数 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：n = 5, limit = 2
 * 输出：3
 * 解释：总共有 3 种方法分配 5 颗糖果，且每位小朋友的糖果数不超过 2 ：(1, 2, 2) ，(2, 1, 2) 和 (2, 2, 1) 。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：n = 3, limit = 3
 * 输出：10
 * 解释：总共有 10 种方法分配 3 颗糖果，且每位小朋友的糖果数不超过 3 ：(0, 0, 3) ，(0, 1, 2) ，(0, 2, 1) ，(0,
 * 3, 0) ，(1, 0, 2) ，(1, 1, 1) ，(1, 2, 0) ，(2, 0, 1) ，(2, 1, 0) 和 (3, 0, 0)
 * 。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= n <= 50
 * 1 <= limit <= 50
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // n 的范围小，枚举就可以
    // 第一个x，第二个y，第三个 n-x-y
    // 注意剪枝，如果分配完x，还有超过2*limit的，跳过
    int distributeCandies(int n, int limit) {
        int ans = 0;
        for (int i = 0; i <= limit; i++) {
            if (n - i > 2 * limit) {
                continue;
            }
            for (int j = 0; j <= limit; j++) {
                if (i + j > n) {
                    continue;
                }
                if (n - i - j <= limit) {
                    ans++;
                }
            }
        }

        return ans;
    }
};
// @lc code=end

