/*
 * @lc app=leetcode.cn id=2462 lang=cpp
 *
 * [2462] 雇佣 K 位工人的总代价
 *
 * https://leetcode.cn/problems/total-cost-to-hire-k-workers/description/
 *
 * algorithms
 * Medium (41.73%)
 * Likes:    95
 * Dislikes: 0
 * Total Accepted:    21.4K
 * Total Submissions: 48.8K
 * Testcase Example:  '[17,12,10,2,7,2,11,20,8]\n3\n4'
 *
 * 给你一个下标从 0 开始的整数数组 costs ，其中 costs[i] 是雇佣第 i 位工人的代价。
 * 
 * 同时给你两个整数 k 和 candidates 。我们想根据以下规则恰好雇佣 k 位工人：
 * 
 * 
 * 总共进行 k 轮雇佣，且每一轮恰好雇佣一位工人。
 * 在每一轮雇佣中，从最前面 candidates 和最后面 candidates
 * 人中选出代价最小的一位工人，如果有多位代价相同且最小的工人，选择下标更小的一位工人。
 * 
 * 比方说，costs = [3,2,7,7,1,2] 且 candidates = 2 ，第一轮雇佣中，我们选择第 4 位工人，因为他的代价最小
 * [3,2,7,7,1,2] 。
 * 第二轮雇佣，我们选择第 1 位工人，因为他们的代价与第 4 位工人一样都是最小代价，而且下标更小，[3,2,7,7,2]
 * 。注意每一轮雇佣后，剩余工人的下标可能会发生变化。
 * 
 * 
 * 如果剩余员工数目不足 candidates 人，那么下一轮雇佣他们中代价最小的一人，如果有多位代价相同且最小的工人，选择下标更小的一位工人。
 * 一位工人只能被选择一次。
 * 
 * 
 * 返回雇佣恰好 k 位工人的总代价。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 输入：costs = [17,12,10,2,7,2,11,20,8], k = 3, candidates = 4
 * 输出：11
 * 解释：我们总共雇佣 3 位工人。总代价一开始为 0 。
 * - 第一轮雇佣，我们从 [17,12,10,2,7,2,11,20,8] 中选择。最小代价是 2 ，有两位工人，我们选择下标更小的一位工人，即第 3
 * 位工人。总代价是 0 + 2 = 2 。
 * - 第二轮雇佣，我们从 [17,12,10,7,2,11,20,8] 中选择。最小代价是 2 ，下标为 4 ，总代价是 2 + 2 = 4 。
 * - 第三轮雇佣，我们从 [17,12,10,7,11,20,8] 中选择，最小代价是 7 ，下标为 3 ，总代价是 4 + 7 = 11 。注意下标为
 * 3 的工人同时在最前面和最后面 4 位工人中。
 * 总雇佣代价是 11 。
 * 
 * 
 * 示例 2：
 * 
 * 输入：costs = [1,2,4,1], k = 3, candidates = 3
 * 输出：4
 * 解释：我们总共雇佣 3 位工人。总代价一开始为 0 。
 * - 第一轮雇佣，我们从 [1,2,4,1] 中选择。最小代价为 1 ，有两位工人，我们选择下标更小的一位工人，即第 0 位工人，总代价是 0 + 1 =
 * 1 。注意，下标为 1 和 2 的工人同时在最前面和最后面 3 位工人中。
 * - 第二轮雇佣，我们从 [2,4,1] 中选择。最小代价为 1 ，下标为 2 ，总代价是 1 + 1 = 2 。
 * - 第三轮雇佣，少于 3 位工人，我们从剩余工人 [2,4] 中选择。最小代价是 2 ，下标为 0 。总代价为 2 + 2 = 4 。
 * 总雇佣代价是 4 。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= costs.length <= 10^5 
 * 1 <= costs[i] <= 10^5
 * 1 <= k, candidates <= costs.length
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 雇佣过程可是用两个最小堆来模拟，一个维护costs剩余数字的最前面candidates个数的最小值
    // 另一个维护costs剩余数字的最后面candidates个数的最小是
    // 具体算法：
    // 1.设 cost 的长度为n。如果candidates*2+k > n，一定可以选到最小的k个数，所以直接放回cost的前k小之和
    // 2.初始化答案ans=0.初始化最小堆pre为costs最前面的candidates个数，初始化最小堆suf为costs最前面的candidates个数。
    //   初始化小标i=candidates，j=n-1-candidates
    // 3.循环k次。每次循环，如果pre的堆顶小于等于suf的堆顶，则弹出pre的堆顶，加入答案，然后把costs[i]加入pre，i加1；
    //   如果suf的堆顶小于pre的堆顶，则弹出suf的堆顶，加入答案，然后把 costs[j]加入suf，j加1
    // 4.返回答案。
    long long totalCost(vector<int>& costs, int k, int candidates) {
        int n = costs.size();
        if (candidates * 2 + k > n) {
            std::sort(costs.begin(), costs.end());
            long long ans = 0;
            for (int i = 0; i < k; i++) {
                ans += costs[i];
            }
            return ans;
        }

        std::priority_queue<int, std::vector<int>, greater<>>  pre, suf;
        for (int i = 0; i < candidates; i++) {
            pre.push(costs[i]);
            suf.push(costs[n - 1 - i]);
        }

        long long ans = 0;
        int i = candidates;
        int j = n - 1 - candidates;
        while (k--) {
            if (pre.top() <= suf.top()) {
                ans += pre.top();
                pre.pop();
                pre.push(costs[i++]);
            } else {
                ans += suf.top();
                suf.pop();
                suf.push(costs[j--]);
            }
        }

        return ans;
    }
};
// @lc code=end

