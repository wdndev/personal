/*
 * @lc app=leetcode.cn id=873 lang=cpp
 *
 * [873] 最长的斐波那契子序列的长度
 *
 * https://leetcode.cn/problems/length-of-longest-fibonacci-subsequence/description/
 *
 * algorithms
 * Medium (56.22%)
 * Likes:    379
 * Dislikes: 0
 * Total Accepted:    52.1K
 * Total Submissions: 92.8K
 * Testcase Example:  '[1,2,3,4,5,6,7,8]'
 *
 * 如果序列 X_1, X_2, ..., X_n 满足下列条件，就说它是 斐波那契式 的：
 * 
 * 
 * n >= 3
 * 对于所有 i + 2 ，都有 X_i + X_{i+1} = X_{i+2}
 * 
 * 
 * 给定一个严格递增的正整数数组形成序列 arr ，找到 arr 中最长的斐波那契式的子序列的长度。如果一个不存在，返回  0 。
 * 
 * （回想一下，子序列是从原序列 arr 中派生出来的，它从 arr 中删掉任意数量的元素（也可以不删），而不改变其余元素的顺序。例如， [3, 5, 8]
 * 是 [3, 4, 5, 6, 7, 8] 的一个子序列）
 * 
 * 
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入: arr = [1,2,3,4,5,6,7,8]
 * 输出: 5
 * 解释: 最长的斐波那契式子序列为 [1,2,3,5,8] 。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入: arr = [1,3,7,11,12,14,18]
 * 输出: 3
 * 解释: 最长的斐波那契式子序列有 [1,11,12]、[3,11,14] 以及 [7,11,18] 。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 3 
 * 
 * 1 
 * 
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    int lenLongestFibSubseq(vector<int>& arr) {
        int arr_size = arr.size();
        std::vector<std::vector<int>> dp(arr_size, std::vector<int>(arr_size, 0));
        int ans = 0;

        // 初始化dp
        for (int i = 0; i < arr_size; i++) {
            for (int j = i + 1; j < arr_size; j++) {
                dp[i][j] = 2;
            }
        }

        std::unordered_map<int, int> idx_map;
        // 降 value:idx映射为哈希表，可以快速通过value获得idx
        for (int idx = 0; idx < arr_size; idx++) {
            idx_map[arr[idx]] = idx;
        }

        for (int i = 0; i < arr_size; i++) {
            for (int j = i + 1; j < arr_size; j++) {
                if (idx_map.count(arr[i] + arr[j])) {
                    // 获取 arr[i] + arr[j] 的 idx，即斐波那契式子序列下一项元素
                    int k = idx_map[arr[i] + arr[j]];

                    dp[j][k] = std::max(dp[j][k], dp[i][j] + 1);
                    ans = std::max(ans, dp[j][k]);
                }
            }
        }

        if (ans >= 3) {
            return ans;
        }
        return 0;
    }
};
// @lc code=end

