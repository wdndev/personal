# 15 动态规划

# 1.杨辉三角

[118. 杨辉三角 - 力扣（LeetCode）](https://leetcode.cn/problems/pascals-triangle/description/)

```C++
给定一个非负整数 numRows，生成「杨辉三角」的前 numRows 行。

在「杨辉三角」中，每个数是它左上方和右上方的数的和。
```

先画个图，可以发现： 从第三行开始：除了第一列，每个位置=上一行右上角位置+上一行上面位置 `dp[i][j] = dp[i-1][j-1] + dp[i-1][j]`

首先初始化整个三角 从第三行开始遍历计算 动态规划等式：`dp[i][j] = dp[i-1][j-1] + dp[i-1][j]`

```C++
class Solution {
public:
    // 注意dp的大小
    vector<vector<int>> generate(int numRows) {

        std::vector<std::vector<int>> dp(numRows);

        for (int i = 0; i < numRows; i++) {
            dp[i].resize(i + 1);
            dp[i][0] = dp[i][i] = 1;
            for (int j = 1; j < i; j++) {
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
            }
        }

        return dp;
    }
};
```
