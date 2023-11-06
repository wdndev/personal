#
# @lc app=leetcode.cn id=115 lang=python3
#
# [115] 不同的子序列
#
# https://leetcode.cn/problems/distinct-subsequences/description/
#
# algorithms
# Hard (51.43%)
# Likes:    1154
# Dislikes: 0
# Total Accepted:    155.4K
# Total Submissions: 301.9K
# Testcase Example:  '"rabbbit"\n"rabbit"'
#
# 给你两个字符串 s 和 t ，统计并返回在 s 的 子序列 中 t 出现的个数，结果需要对 10^9 + 7 取模。
# 
# 
# 
# 示例 1：
# 
# 
# 输入：s = "rabbbit", t = "rabbit"
# 输出：3
# 解释：
# 如下所示, 有 3 种可以从 s 中得到 "rabbit" 的方案。
# rabbbit
# rabbbit
# rabbbit
# 
# 示例 2：
# 
# 
# 输入：s = "babgbag", t = "bag"
# 输出：5
# 解释：
# 如下所示, 有 5 种可以从 s 中得到 "bag" 的方案。 
# babgbag
# babgbag
# babgbag
# babgbag
# babgbag
# 
# 
# 
# 
# 提示：
# 
# 
# 1 <= s.length, t.length <= 1000
# s 和 t 由英文字母组成
# 
# 
#

# @lc code=start
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        n = len(s)
        m = len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for j in range(n + 1):
            dp[0][j] = 1
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if t[i - 1] == s[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]  + dp[i][j - 1]
                else:
                    dp[i][j] = dp[i][j - 1]
        #print(dp)
        return dp[m][n]

# @lc code=end

