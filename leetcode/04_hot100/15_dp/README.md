# 15 动态规划

# 1.爬楼梯

[70. 爬楼梯 - 力扣（LeetCode）](https://leetcode.cn/problems/climbing-stairs/description/?envType=study-plan-v2\&envId=top-100-liked "70. 爬楼梯 - 力扣（LeetCode）")

```json
假设你正在爬楼梯。需要 n 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

 

示例 1：

输入：n = 2
输出：2
解释：有两种方法可以爬到楼顶。
1. 1 阶 + 1 阶
2. 2 阶
```

1.  递归，超时
2.  递归 + 哈希cache
3.  dp

```c++
class Solution {
public:
    // 1.递归， 超时
    int climbStairs1(int n) {
        if (n <= 2) {
            return n;
        }

        return this->climbStairs(n - 1) + this->climbStairs(n - 2);
    }

    // 2.递归 + 哈希cache
    int climbStairs2(int n) {
        if (n <= 2) {
            return n;
        }

        std::vector<int> memo(n+1, 0);
        return this->recur(n, memo);
    }

    int recur(int n, std::vector<int>& memo) {
        if (n <= 2) {
            return n;
        }

        if (memo[n] == 0) {
            memo[n] = this->recur(n - 1, memo) + this->recur(n - 2, memo);
        }

        return memo[n];
    }

    // 3.动态规划
    int climbStairs(int n) {
        if (n <= 2) {
            return n;
        }

        std::vector<int> dp(n + 1, 0);
        dp[0] = 0;
        dp[1] = 1;
        dp[2] = 2;

        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }

        return dp[n];
    }

};
```

# 2.杨辉三角

[118. 杨辉三角 - 力扣（LeetCode）](https://leetcode.cn/problems/pascals-triangle/description/ "118. 杨辉三角 - 力扣（LeetCode）")

```c++
给定一个非负整数 numRows，生成「杨辉三角」的前 numRows 行。

在「杨辉三角」中，每个数是它左上方和右上方的数的和。
```

先画个图，可以发现： 从第三行开始：除了第一列，每个位置=上一行右上角位置+上一行上面位置 `dp[i][j] = dp[i-1][j-1] + dp[i-1][j]`

首先初始化整个三角 从第三行开始遍历计算 动态规划等式：`dp[i][j] = dp[i-1][j-1] + dp[i-1][j]`

```c++
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

# 3.打家劫舍

[198. 打家劫舍 - 力扣（LeetCode）](https://leetcode.cn/problems/house-robber/description/?envType=study-plan-v2\&envId=top-100-liked "198. 打家劫舍 - 力扣（LeetCode）")

```json
你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。
```

DP方法：

1.  分治子问题：
2.  状态数组定义：
3.  DP方程：

分析：

-   `a[i]` ： 0\~i 能偷盗的最大数量，结果为`a[n-1]`
-   `a[i][0, 1]` ： 增加一个维度，其中，0表示i偷，1表示i不偷
-   $a[i][0] = max(a[i-1][0], a[i-1][1])$：当前不偷，前一个偷还是不偷的最大值
-   $a[i][1] = a[i-1][0] + nums[i]$ ： 当前偷，等于前一个的值 + 当前的值

```c++
class Solution {
public:
    // 二维动态规划
    int rob(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) {
            return 0;
        }

        // 增加一个维度，其中，0表示i偷，1表示i不偷
        std::vector<std::vector<int>> dp(n, std::vector<int>(2, 0));

        dp[0][0] = 0;
        dp[0][1] = nums[0];

        for (int i = 1; i < n; i++) {
            // 当前不偷，前一个偷还是不偷的最大值
            dp[i][0] = std::max(dp[i - 1][0], dp[i - 1][1]);
            //  当前偷，等于前一个的值 + 当前的值
            dp[i][1] = dp[i - 1][0] + nums[i];
        }

        return std::max(dp[n-1][0], dp[n-1][1]);
    }
};
```

进一步简化：

-   `a[i]` ： 0\~i 能偷盗的最大数量，结果为`max(a)`
-   `a[i]` ： 0\~i 能偷盗的最大数量，且`nums[i] `必须偷的最大值
-   $a[i] = max(a[i-1] + 0, a[i-2] + nums[i])$：当前的最大值等于，上一次偷的最大值 + 0(今天不偷) 和 上上一次偷的最大值 + 偷今天

```c++
class Solution {
public:
    // 简化为一维动态规划
    int rob(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) {
            return 0;
        }
        if (n == 1) {
            return nums[0];
        }

        std::vector<int> dp(n,0);

        dp[0] = nums[0];
        dp[1] = std::max(nums[0], nums[1]);

        int res = dp[1];

        for (int i = 2; i < n; i++) {
            dp[i] = std::max(dp[i - 1] + 0, dp[i - 2] + nums[i]);

            res = std::max(res, dp[i]);
        }

        return res;
    }
};
```

# 4.完全平方数

[279. 完全平方数 - 力扣（LeetCode）](https://leetcode.cn/problems/perfect-squares/description/?envType=study-plan-v2\&envId=top-100-liked "279. 完全平方数 - 力扣（LeetCode）")

```json
给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。

完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。

示例 1：

输入：n = 12
输出：3 
解释：12 = 4 + 4 + 4
```

首先初始化长度为 n+1 的数组 dp，每个位置都为 0

对数组进行遍历，下标为 i，每次都将当前数字先更新为最大的结果，即 `dp[i]=i`，比如 i=4，最坏结果为 4=1+1+1+1 即为 4 个数字

动态转移方程为：`dp[i] = MIN(dp[i], dp[i - j * j] + 1)`，`i `表示当前数字，`j*j` 表示平方数

-   状态定义：`f[i] `表示最少需要多少个数的平方来表示整数 i。
-   转移方程：`f[i] = MIN(f[i], f[i - j * j] + 1)`，其中，$1<j<\sqrt i$，

```java
class Solution {
public:
    int numSquares(int n) {
        std::vector<int> dp(n+1, 0);

        for (int i = 1; i <= n; i++) {
            // 最坏的情况就是每次+1
            dp[i] = i;
            for (int j = 1; i - j * j >= 0; j ++) {
                // 动态转移方程
                dp[i] = std::min(dp[i], dp[i - j *j] + 1);
            }
        }

        return dp[n];
    }
};

```

# 5.零钱兑换

[322. 零钱兑换 - 力扣（LeetCode）](https://leetcode.cn/problems/coin-change/description/?envType=study-plan-v2\&envId=top-100-liked "322. 零钱兑换 - 力扣（LeetCode）")

```json
给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。

计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。

你可以认为每种硬币的数量是无限的。
```

DP方法

1.  分治（子问题）：$f[n] = min~ \{f(n-k), for ~k ~in [1, 2, 5]\} + 1$
2.  状态数组定义：$f(n)$
3.  DP方程：$f[n] = min ~\{f(n-k), for ~k ~in [1, 2, 5]\} + 1$

```c++
class Solution {
public:
    // 1.动态规划
    int coinChange(vector<int>& coins, int amount) {
        int max_amount = amount + 1;
        std::vector<int> dp(amount + 1, max_amount);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int j = 0; j < coins.size(); j++) {
                if (coins[j] <= i) {
                    dp[i] = std::min(dp[i], dp[i - coins[j]] + 1);
                }
            }
        }

        return dp[amount] > amount ? -1 : dp[amount];
    }
};
```

# 6.单词拆分

[139. 单词拆分 - 力扣（LeetCode）](https://leetcode.cn/problems/word-break/description/?envType=study-plan-v2\&envId=top-100-liked "139. 单词拆分 - 力扣（LeetCode）")

```c++
给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s 。

注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。


示例 1：

输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成。
```

1.  动态规划
    1.  状态空间 `dp[i]`，s的前i位是否可以用wordDict中的单词表示，dp长度为 n+1，初始`dp[0]=true`，表示空字符可以被表示
    2.  遍历字符s，判断每一个单词是否在s的前i个字符中，即`dp[i]`是否为true。
2.  记忆化搜索
    1.  若s长度为0，则返回true，表示wordDict中的单词已分割完
    2.  初始化当前字符串是否可以被分割 res=false
    3.  遍历字符串s，遍历区间为`[1, n+1]`。若`s[0,..., i-1]`在wordDict中：$res = backtrack(s[i,..., n-1]) or res$。解释：保存遍历结束索引中，可以使字符串切割完成的情况。
3.  返回`backtrack(s)`

```c++
class Solution {
public:
    // 1.动态规划
    bool wordBreak(string s, vector<string>& wordDict) {
        int str_len = s.size();
        // 通过看能否通过修改让最后一个元素的值为true，如果可以，则返回true，否则返回false;
        std::vector<bool> dp(str_len + 1, false);
        dp[0] = true;

        // 遍历字符串s
        for (int i = 0; i < str_len; i++) {
            // 如果前i位不能表示为word中的某一个单词，则跳过
            if (!dp[i]) {
                continue;
            }
            // 遍历单词
            // 此处有for循环，可能进行多处修改，修改为true地方我们后续当i遍历到该位置时要继续进行判定，
            // 如果有一次连续修改使得dp[len]成功改成了true,则直接break并返回true；
            // 也就是说，i遍历到每个dp[i]==true的位置，都有机会将dp[len]修改为true，
            // 如果遍历完全都没能把dp[len]修改为true，则说明无法成功拼接出字符串s
            for (auto& word : wordDict) {
                if (word.size() + i <= str_len && s.substr(i, word.size()) == word) {
                    dp[i + word.size()] = true;
                }
            }
        }
        return dp[str_len];
    }

    // 2.记忆化搜索 + dfs
    bool wordBreak2(string s, vector<string>& wordDict) {
        std::unordered_set<std::string> uset;
        for (auto& w : wordDict) {
            uset.insert(w);
        }
        // 使用一个数组来记录从每个索引位置开始的子问题是否可解
        std::vector<int> memo(s.size(), 0);

        return this->dfs(s, uset, 0, memo);
    }

    bool dfs(std::string s, std::unordered_set<std::string>& uset, int idx, std::vector<int>& memo) {
        if (idx >= s.size()) {
            return true;
        }

        if (memo[idx] != 0) {
            return memo[idx] == 1;
        }

        std::string tmp_str = "";
        for (int i = idx; i < s.size(); i++) {
            tmp_str = tmp_str + s[i];
            // 如果字典包含当前的词语，且切割下去可行，标记为true，并返回
            if (uset.find(tmp_str) != uset.end() && this->dfs(s, uset, i + 1, memo)) {
                memo[idx] = 1;
                return true;
            }
        }

        // 标记从当前位置切割不行
        memo[idx] = -1;

        return false;
    }
};
```

# 7.最长递增子序列

[300. 最长递增子序列 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-increasing-subsequence/description/?envType=study-plan-v2\&envId=top-100-liked "300. 最长递增子序列 - 力扣（LeetCode）")

```c++
给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

子序列 是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

示例 1：

输入：nums = [10,9,2,5,3,7,101,18]
输出：4
解释：最长递增子序列是 [2,3,7,101]，因此长度为 4 。

示例 2：

输入：nums = [0,1,0,3,2,3]
输出：4
```

1.  动态规划
    1.  状态定义：`dp[i]` ：前i个元素中的最长上升序列
    2.  状态转移方程：`dp[i] = max(dp[i], dp[j] + 1) `，其中` 0≤j<i且nums[j] < nums[i]`
2.  贪心 + 二分查找：设当前已求出的最长上升子序列`d`的长度为 `len`（初始时为 1），从前往后遍历数组 `nums`，在遍历到 `nums[i]`时：
    1.  如果 $ nums[i]>d[len]  $，则直接加入到 d 数组末尾，并更新 `len=len+1`；
    2.  否则，在 d 数组中二分查找，找到第一个比` nums[i]`小的数 `d[k]`，并更新 `d[k+1]=nums[i]`

以输入序列 `[0,8,4,12,2] `为例：

-   第一步插入 0，d=\[0]；
-   第二步插入 8，d=\[0,8]；
-   第三步插入 4，d=\[0,4]；
-   第四步插入 12，d=\[0,4,12]；
-   第五步插入 2，d=\[0,2,12]。

最终得到最大递增子序列长度为 3。

```c++
class Solution {
public:
    // 1.动态规划
    int lengthOfLIS1(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) {
            return 0;
        }

        int max_len = 1;

        std::vector<int> dp(n, 0);
        for (int i = 0; i < n; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    dp[i] = std::max(dp[i], dp[j] + 1);
                    max_len = std::max(max_len, dp[i]);
                }
            }
        }

        return max_len;
    }

    // 2.贪心+ 二分查找
    int lengthOfLIS(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) {
            return 0;
        }

        // 当前已求出的最长上升子序列d的长度为 len（初始时为 1）
        std::vector<int> d(n + 1, 0);
        int len = 1;
        d[len] = nums[0];

        // 遍历nums
        for (int i = 1; i < n; i++) {
            // 如果 nums[i]>d[len] ，则直接加入到 d 数组末尾，并更新 len=len+1；
            // 否则，在 d 数组中二分查找，找到第一个比 nums[i]小的数 d[k]，
            // 并更新 d[k+1]=nums[i]
            if (nums[i] > d[len]) {
                len++;
                d[len] = nums[i];
            } else {
                int left = 1;
                int right = len;
                int pos = 0;
            
                while (left <= right) {
                    int mid = left + (right - left) / 2;
                    if (d[mid] < nums[i]) {
                        pos = mid;
                        left = mid + 1;
                    } else {
                        right = mid - 1;
                    }
                }
                d[pos + 1] = nums[i];
            }
        }

        return len;
    }
};
```

# 8.乘积最大子数组

[152. 乘积最大子数组 - 力扣（LeetCode）](https://leetcode.cn/problems/maximum-product-subarray/description/?envType=study-plan-v2\&envId=top-100-liked "152. 乘积最大子数组 - 力扣（LeetCode）")

```c++
给你一个整数数组 nums ，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

测试用例的答案是一个 32-位 整数。

子数组 是数组的连续子序列。


示例 1:

输入: nums = [2,3,-2,4]
输出: 6
解释: 子数组 [2,3] 有最大乘积 6。
```

动态规划：

-   遍历数组时计算当前最大值，不断更新
-   令`max_value`为当前最大值，则当前最大值为 `max_value=max(max_value* nums[i], nums[i])`
-   由于存在负数，那么会导致最大的变最小，最小的变最大。因此还需要维护当前最小值 `min_value`，`min_value = min(min_value * nums[i], nums[i])`
-   当出现负数时，`max_value`和`min_value`进行交换后再计算

```c++
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int max_value = nums[0];
        int min_value = nums[0];
        int ans = nums[0];

        for (int i = 1; i < nums.size(); i++) {
            // 由于存在负数，那么会导致最大的变最小，最小的变最大。
            if (nums[i] < 0) {
                int tmp = max_value;
                max_value = min_value;
                min_value = tmp;
            }

            max_value = std::max(max_value * nums[i], nums[i]);
            min_value = std::min(min_value * nums[i], nums[i]);

            ans = std::max(ans, max_value);
        }

        return ans;
    }
};
```

# 9.分割等和子集

[416. 分割等和子集 - 力扣（LeetCode）](https://leetcode.cn/problems/partition-equal-subset-sum/description/?envType=study-plan-v2\&envId=top-100-liked "416. 分割等和子集 - 力扣（LeetCode）")

```c++
给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

 

示例 1：

输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11] 。

示例 2：

输入：nums = [1,2,3,5]
输出：false
解释：数组不能分割成两个元素和相等的子集。

```

画一个 `len `行，`target + 1` 列的表格。这里 `len `是物品的个数，target 是背包的容量。`len `行表示一个一个物品考虑，`target + 1`多出来的那 1 列，表示背包容量从 0 开始考虑。很多时候，我们需要考虑这个容量为 0 的数值。

**状态定义**：`dp[i][j]`表示从数组的 `[0, i]` 这个子区间内挑选一些正整数，每个数只能用一次，使得这些数的和恰好等于` j`。
**状态转移方程**：很多时候，状态转移方程思考的角度是「分类讨论」，对于「0-1 背包问题」而言就是「当前考虑到的数字选与不选」。

-   不选择 `nums[i]`，如果在 `[0, i - 1]` 这个子区间内已经有一部分元素，使得它们的和为 `j` ，那么 `dp[i][j] = true`；
-   选择 `nums[i]`，如果在 `[0, i - 1]` 这个子区间内就得找到一部分元素，使得它们的和为` j - nums[i]。`

状态转移方程：

$$
dp[i][j] = dp[i - 1][j] ~or ~dp[i - 1][j - nums[i]]
$$

一般写出状态转移方程以后，就需要考虑初始化条件。

-   `j - nums[i] `作为数组的下标，一定得保证大于等于 0 ，因此 `nums[i] <= j`；
-   注意到一种非常特殊的情况：`j` 恰好等于 `nums[i]`，即单独 `nums[i]` 这个数恰好等于此时「背包的容积」 `j`，这也是符合题意的。

因此完整的状态转移方程是

$$
\mathrm{dp}[i][j]=\left\{\begin{array}{ll}\mathrm{dp}[i-1][j], & \text { 至少是这个答案, 如果 } \mathrm{dp}[i-1][j] \text { 为真, 直接计算下一个状态 } \\ \operatorname{truc}, & \text { nums }[\mathrm{i}]=\mathrm{j} \\ \mathrm{dp}[i-1][j-\text { nums }[i]] . & \text { nums }[\mathrm{i}]<\mathrm{j}\end{array}\right.
$$

说明：虽然写成花括号，但是它们的关系是 **或者** 。

-   初始化：`dp[0][0] = false`，因为候选数 `nums[0]` 是正整数，凑不出和为 0；
-   输出：`dp[len - 1][target]`，这里 len 表示数组的长度，target 是数组的元素之和（必须是偶数）的一半。

```c++
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int len = nums.size();

        // 求和
        int sum = 0;
        for (auto& n : nums) {
            sum += n;
        }
        // 判断奇偶数，如果是奇数，直接返回，
        if ((sum & 1) == 1) {
            return false;
        }

        int target = sum / 2;
        // 创建二维状态数组，行：物品索引，列：容量（包括0）
        std::vector<std::vector<bool>> dp(len, std::vector<bool>(target + 1, false));

        // 先填表格第0行，第1个数只能让容积为它子集的背包恰好装满
        if (nums[0] <= target) {
            dp[0][nums[0]] = true;
        }
        // 再填表格后面几行
        for (int i = 1; i < len; i++) {
            for (int j = 0; j <= target; j++) {
                // 直接从上一行先把结果抄下来，然后修正
                dp[i][j] = dp[i - 1][j];

                if (nums[i] == j) {
                    dp[i][j] = true;
                    continue;
                }
                if (nums[i] < j) {
                    dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i]];
                }
            }
        }

        return dp[len - 1][target];
    }
};
```

# 10.最长有效括号

[32. 最长有效括号 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-valid-parentheses/description/?envType=study-plan-v2\&envId=top-100-liked "32. 最长有效括号 - 力扣（LeetCode）")

```c++
给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。


示例 1：

输入：s = "(()"
输出：2
解释：最长有效括号子串是 "()"
示例 2：

输入：s = ")()())"
输出：4
解释：最长有效括号子串是 "()()"
示例 3：

输入：s = ""
输出：0
```

定义` dp[i]` 表示以下标 i 字符结尾的最长有效括号的长度。将 `dp `数组全部初始化为 0 。显然有效的子串一定以 `)` 结尾，因此可以知道以 `( `结尾的子串对应的 `dp `值必定为 0 ，需要求解 `)` 在 dp 数组中对应位置的值。

从前往后遍历字符串求解 `dp `值，每两个字符检查一次：

1.  `s[i]=‘)’` 且 `s[i−1]=‘(’`，也就是字符串形如 `“……()”`，可以推出：`dp[i]=dp[i−2]+2`
    ，可以进行这样的转移，是因为结束部分的 "()" 是一个有效子字符串，并且将之前有效子字符串的长度增加了 2 。
2.  `s[i]=‘)’`且 `s[i−1]=‘)’`，也就是字符串形如 `“……))”`，可以推出：如果 `s[i−dp[i−1]−1]=‘(’`，那么`dp[i]=dp[i−1]+dp[i−dp[i−1]−2]+2`

```c++
class Solution {
public:
    // 1.动态规划
    int longestValidParentheses(string s) {
        int n = s.size();
        // 示以下标 i 字符结尾的最长有效括号的长度
        std::vector<int> dp(n, 0);
        int max_ans = 0;

        for (int i = 1; i < n; i++) {
            if (s[i] == ')') {
                // ……()形式，
                if (s[i - 1] == '(') {
                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                // ……))形式，s[i−dp[i−1]−1]=‘(’为寻找和当前）匹配的括号
                } else if (i - dp[i - 1] > 0 && s[i - dp[i - 1] - 1] == '(') {
                    dp[i] = dp[i - 1] + ((i - dp[i - 1]) >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;
                }
                max_ans = std::max(max_ans, dp[i]);
            }
        }

        return max_ans;
    }
};
```
