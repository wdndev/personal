/*
 * @lc app=leetcode.cn id=1997 lang=cpp
 *
 * [1997] 访问完所有房间的第一天
 *
 * https://leetcode.cn/problems/first-day-where-you-have-been-in-all-the-rooms/description/
 *
 * algorithms
 * Medium (35.56%)
 * Likes:    125
 * Dislikes: 0
 * Total Accepted:    15.5K
 * Total Submissions: 35.1K
 * Testcase Example:  '[0,0]'
 *
 * 你需要访问 n 个房间，房间从 0 到 n - 1 编号。同时，每一天都有一个日期编号，从 0 开始，依天数递增。你每天都会访问一个房间。
 * 
 * 最开始的第 0 天，你访问 0 号房间。给你一个长度为 n 且 下标从 0 开始 的数组 nextVisit 。在接下来的几天中，你访问房间的 次序
 * 将根据下面的 规则 决定：
 * 
 * 
 * 假设某一天，你访问 i 号房间。
 * 如果算上本次访问，访问 i 号房间的次数为 奇数 ，那么 第二天 需要访问 nextVisit[i] 所指定的房间，其中 0 <=
 * nextVisit[i] <= i 。
 * 如果算上本次访问，访问 i 号房间的次数为 偶数 ，那么 第二天 需要访问 (i + 1) mod n 号房间。
 * 
 * 
 * 请返回你访问完所有房间的第一天的日期编号。题目数据保证总是存在这样的一天。由于答案可能很大，返回对 10^9 + 7 取余后的结果。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nextVisit = [0,0]
 * 输出：2
 * 解释：
 * - 第 0 天，你访问房间 0 。访问 0 号房间的总次数为 1 ，次数为奇数。
 * 下一天你需要访问房间的编号是 nextVisit[0] = 0
 * - 第 1 天，你访问房间 0 。访问 0 号房间的总次数为 2 ，次数为偶数。
 * 下一天你需要访问房间的编号是 (0 + 1) mod 2 = 1
 * - 第 2 天，你访问房间 1 。这是你第一次完成访问所有房间的那天。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nextVisit = [0,0,2]
 * 输出：6
 * 解释：
 * 你每天访问房间的次序是 [0,0,1,0,0,1,2,...] 。
 * 第 6 天是你访问完所有房间的第一天。
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：nextVisit = [0,1,2,0]
 * 输出：6
 * 解释：
 * 你每天访问房间的次序是 [0,0,1,1,2,2,3,...] 。
 * 第 6 天是你访问完所有房间的第一天。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * n == nextVisit.length
 * 2 <= n <= 10^5
 * 0 <= nextVisit[i] <= i
 * 
 * 
 */

// @lc code=start
class Solution {
public: 
    // 来源：https://leetcode.cn/problems/first-day-where-you-have-been-in-all-the-rooms/
    // 根据题意，首次访问房间i时，因为时第一次访问，1是奇数，所以下一天一定要访问 j= nextVisit[i] 房间，下文简称回访
    // 由于访问偶数次，才能访问右边的下一个房间，所以对于i左边的房间，我们一定都访问了偶数次
    // 对于房间i，其访问次数的奇偶性变化如下：访问房间i之前：偶数；访问到房间i：奇数；回访完毕，重新回到房间i：偶数
    // 1.状态定义：f[i]表示：从 [访问房间i且次数为奇数]到[访问到房间i且次数为偶数]所需要的天数
    // 2.状态转移方程：由于 [j,i-1]范围内的每个房间都需要「回访」，所以需要把这个范围内的 f 值都加起来，再算上房间 i 需要访问 2 次，
    //     于是得到如下状态转移方程： f[i] = 2 + sum_{k=j}^{i-1}(f[k])
    // 3.状态使用前缀和优化：s[i+1] = s[i] * 2 - s[j] + 2

    int firstDayBeenInAllRooms(vector<int>& nextVisit) {
        const int mod = 1e9 + 7;
        int n = nextVisit.size();
        std::vector<long> s(n);
        for (int i = 0; i < n - 1; i++) {
            int j = nextVisit[i];
            // + MOD 避免算出负数
            s[i + 1] = (s[i] * 2 - s[j] + 2 + mod) % mod;
        }
        return s[n - 1];
    }
};
// @lc code=end

