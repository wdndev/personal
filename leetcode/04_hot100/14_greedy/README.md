# 14 greedy

# 1.买卖股票的最佳时机

[121. 买卖股票的最佳时机 - 力扣（LeetCode）](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/description/?envType=study-plan-v2\&envId=top-100-liked "121. 买卖股票的最佳时机 - 力扣（LeetCode）")

```c++
给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。

你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。

 

示例 1：

输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
```

1.  暴力法：找出数组中两个数字之间的最大差值，并卖出的价格必须大于买入的价格
2.  一次遍历：维护一个当前最小值和最大利润

```c++
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
```

# 2.跳跃游戏

[55. 跳跃游戏 - 力扣（LeetCode）](https://leetcode.cn/problems/jump-game/description/?envType=study-plan-v2\&envId=top-100-liked "55. 跳跃游戏 - 力扣（LeetCode）")

```c++
给你一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标，如果可以，返回 true ；否则，返回 false 。


示例 1：

输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
```

1.  暴力求解：O(n^2)，超时，使用额外的一个数组，初始化为false，依据不同的下标，进行跳跃，同时修改数组位置为true，最后观察数组最后一个位置是不是true
2.  贪心算法：从后往前贪心

```c++
class Solution {
public:
    // 1.暴力求解O(n^2)， 超时
    // 使用额外的一个数组，初始化为false，依据不同的下标，进行跳跃，
    // 同时修改数组位置为true，最后观察数组最后一个位置是不是true
    bool canJump1(vector<int>& nums) {
        if (nums.size() == 0) {
            return false;
        }
        std::vector<int> flag_arr(nums.size(), 0);
        if (nums[0] == 0 && nums.size() > 1) {
            return false;
        }
        flag_arr[0] = 1;
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] == 0) {
                continue;
            }
            for (int j = i + 1; j <= i + nums[i]; j++) {
                if (j < nums.size()) {
                    flag_arr[j] = 1;
                }
            }
        }

        return flag_arr[nums.size() - 1] == 1;
    }

    // 2.贪心算法：从后往前贪心
    bool canJump(vector<int>& nums) {
        if (nums.size() == 0) {
            return false;
        }

        int end_reachable = nums.size() - 1;

        for (int i = nums.size() - 1; i >= 0; i--) {
            if (nums[i] + i >= end_reachable) {
                end_reachable = i;
            }
        }

        return end_reachable == 0;
    } 
};
```

# 3.跳跃游戏Ⅱ

[45. 跳跃游戏 II - 力扣（LeetCode）](https://leetcode.cn/problems/jump-game-ii/description/?envType=study-plan-v2\&envId=top-100-liked "45. 跳跃游戏 II - 力扣（LeetCode）")

```c++
给定一个长度为 n 的 0 索引整数数组 nums。初始位置为 nums[0]。

每个元素 nums[i] 表示从索引 i 向前跳转的最大长度。换句话说，如果你在 nums[i] 处，你可以跳转到任意 nums[i + j] 处:

- 0 <= j <= nums[i] 
- i + j < n

返回到达 nums[n - 1] 的最小跳跃次数。生成的测试用例可以到达 nums[n - 1]。

示例 1:

输入: nums = [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
```

1.  如果某一个作为起跳点的格子可以跳跃距离是3，那么表示后面3个格子都可以作为起跳点。可以对每一个能作为起跳点的格子都尝试跳一次，把能跳到最远的距离不断更新；
2.  如果从这个起跳点起跳叫做第一次跳跃，那么从后面3个格子起跳都可以叫做第2次跳跃。
3.  所以，当一次跳跃结束时，从下一个格子开始，到现在能跳到最远的距离，都是下一次跳跃的起跳点；
    1.  对每一次跳跃，用for循环来模拟
    2.  跳完一次后，更新下一个起跳点的范围
    3.  在新的范围内跳，更新能跳到最远的距离
4.  记录跳跃的次数，如果跳跃到终点，就得到了结果

```c++
class Solution {
public:
    int jump(vector<int>& nums) {
        int ans = 0;
        int start = 0;
        int end = 1;

        while (end < nums.size()) {
            int max_pos = 0;
            for (int i = start; i < end; i++) {
                // 能跳到的最远距离
                max_pos = std::max(max_pos, i + nums[i]);
            }

            // 更新位置
            // 下次起跳点范围为开始的格子
            start = end;
            // 下次起跳点范围结束的格子
            end = max_pos + 1;

            ans++;
        }

        return ans;
    }
};
```

# 4.划分字母区间

[763. 划分字母区间 - 力扣（LeetCode）](https://leetcode.cn/problems/partition-labels/description/?envType=study-plan-v2\&envId=top-100-liked "763. 划分字母区间 - 力扣（LeetCode）")

```c++
给你一个字符串 s 。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。

注意，划分结果需要满足：将所有划分结果按顺序连接，得到的字符串仍然是 s 。

返回一个表示每个字符串片段的长度的列表。


示例 1：
输入：s = "ababcbacadefegdehijhklij"
输出：[9,7,8]
解释：
划分结果为 "ababcbaca"、"defegde"、"hijhklij" 。
每个字母最多出现在一个片段中。
像 "ababcbacadefegde", "hijhklij" 这样的划分是错误的，因为划分的片段数较少。 
```

由于同一个字母只能出现在同一个片段，显然同一个字母的第一次出现的下标位置和最后一次出现的下标位置必须出现在同一个片段。因此需要遍历字符串，**得到每个字母最后一次出现的下标位置**。

在得到每个字母最后一次出现的下标位置之后，可以**使用贪心的方法将字符串划分为尽可能多的片段**，具体做法如下。

1.  从左到右遍历字符串，遍历的同时维护当前片段的开始下标 `start `和结束下标 `end`，初始时 `start=end=0`。
2.  对于每个访问到的字母 `c`，得到当前字母的最后一次出现的下标位置 `end_c`，则当前片段的结束下标一定不会小于 `end_c`，因此令 `end=max⁡(end,end_c)`
3.  当访问到下标 `end`时，当前片段访问结束，当前片段的下标范围是 `[start,end]`，长度为 `end−start+1`，将当前片段的长度添加到返回值，然后令 `start=end+1`，继续寻找下一个片段。
4.  重复上述过程，直到遍历完字符串。

上述做法**使用贪心的思想寻找每个片段可能的最小结束下标**，因此可以保证每个片段的长度一定是符合要求的最短长度，如果取更短的片段，则一定会出现同一个字母出现在多个片段中的情况。由于每次取的片段都是符合要求的最短的片段，因此得到的片段数也是最多的。

由于每个片段访问结束的标志是访问到下标 `end`，因此对于每个片段，可以保证当前片段中的每个字母都一定在当前片段中，不可能出现在其他片段，可以保证同一个字母只会出现在同一个片段。

```c++
class Solution {
public:
    vector<int> partitionLabels(string s) {
        // 记录每个字符出现在数组中的最后一个位置
        int last_char_pos[26];
        int str_len = s.size();
        for (int i = 0; i < str_len; i++) {
            last_char_pos[s[i] - 'a'] = i;
        }

        std::vector<int> ans;
        int start = 0;
        int end = 0;

        for (int i = 0; i < str_len; i++) {
            end = std::max(end, last_char_pos[s[i] - 'a']);
            if (end == i) {
                ans.push_back(end - start + 1);
                start = end + 1;
            }
        }

        return ans;
    }
};
```
