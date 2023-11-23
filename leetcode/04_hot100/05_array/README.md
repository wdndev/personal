# 05 Array

# 1.最大子数组和

[53. 最大子数组和 - 力扣（LeetCode）](https://leetcode.cn/problems/maximum-subarray/description/?envType=study-plan-v2\&envId=top-100-liked "53. 最大子数组和 - 力扣（LeetCode）")

```bash
给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

子数组 是数组中的一个连续部分。

示例 1：

输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。

```

1.  DP求解：
    1.  分治（子问题）：如果第i个元素，则子序列和是多少？$max\_sum(i) = Max(max\_sum(i-1), 0) + a[i]$
    2.  状态数组定义：$f[1]$
    3.  DP方程：$f(i) = Max(f(i-1), 0) + a[i]$

```c++
class Solution {
public:
    // 1. DP求解：
    //     1. 分治（子问题）：如果第i个元素，则子序列和是多少？$max_sum(i) = Max(max_sum(i-1), 0) + a[i]$
    //     2. 状态数组定义：$f[1]$
    //     3. DP方程：$f(i) = Max(f(i-1), 0) + a[i]$
    int maxSubArray(vector<int>& nums) {
        std::vector<int> dp(nums);
        int max_sum = dp[0];

        for (int i = 1; i < nums.size(); i++) {
            dp[i] = std::max(dp[i], dp[i] + dp[i - 1]);
            max_sum = dp[i] >= max_sum ? dp[i] : max_sum;
        }

        return max_sum;
    }
};
```

# 2.合并区间

[56. 合并区间 - 力扣（LeetCode）](https://leetcode.cn/problems/merge-intervals/description/?envType=study-plan-v2\&envId=top-100-liked "56. 合并区间 - 力扣（LeetCode）")

```bash
以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。


示例 1：

输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
```

排序：如果我们按照区间的左端点排序，那么在排完序的列表中，可以合并的区间一定是连续的。

用数组 merged 存储最终的答案。

-   首先，将列表中的区间按照左端点升序排序。然后将第一个区间加入 merged 数组中，并按顺序依次考虑之后的每个区间：
-   如果当前区间的左端点在数组 merged 中最后一个区间的右端点之后，那么它们不会重合，可以直接将这个区间加入数组 merged 的末尾；
-   否则，它们重合，需要用当前区间的右端点更新数组 merged 中最后一个区间的右端点，将其置为二者的较大值。

```c++
class Solution {
public:
    // 左端点排序处理
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        if (intervals.size() == 0) {
            return {};
        }
        // 左端点排序
        std::sort(intervals.begin(), intervals.end());

        // 合并数组结果
        std::vector<std::vector<int>> merged;

        // 遍历数组
        for (int i = 0; i < intervals.size(); i++) {
            int left = intervals[i][0];
            int right = intervals[i][1];

            // 如果merged为空，或是merged最后一个的右端点小于当前的左端点，加入数组
            // 佛则，进行比较，选择最大的
            if (!merged.size() || merged.back()[1] < left) {
                merged.push_back({left, right});
            } else {
                merged.back()[1] = std::max(merged.back()[1], right);
            }
        }

        return merged;
    }
```

# 3.轮转数组

[189. 轮转数组 - 力扣（LeetCode）](https://leetcode.cn/problems/rotate-array/description/?envType=study-plan-v2\&envId=top-100-liked "189. 轮转数组 - 力扣（LeetCode）")

```bash
给定一个整数数组 nums，将数组中的元素向右轮转 k 个位置，其中 k 是非负数。

示例 1:

输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右轮转 1 步: [7,1,2,3,4,5,6]
向右轮转 2 步: [6,7,1,2,3,4,5]
向右轮转 3 步: [5,6,7,1,2,3,4]
```

翻转三次数组即可。先翻转整个数组，后翻转前k个数字，最后翻转后nums.size()-k个数字

```c++
class Solution {
public:
    // 我们可以先将所有元素翻转，这样尾部的 k mod n 个元素就被移至数组头部，
    // 然后我们再翻转 [0,k mod n−1]区间的元素和 [k mod n,n−1] 区间的元素即能得到最后的答案。
    void rotate(vector<int>& nums, int k) {
        // 如果数组大小为1，直接返回
        if (nums.size() == 1) {
            return;
        }

        k = k % nums.size();

        // 翻转全部数组
        this->reverse(nums, 0, nums.size() - 1);
        // 翻转前k个
        this->reverse(nums, 0, k - 1);
        // 翻转后面的
        this->reverse(nums, k, nums.size() - 1);
    }

    // 翻转数组
    void reverse(std::vector<int>& nums, int start, int end) {
        int tmp = 0;
        while (start < end) {
            tmp = nums[start];
            nums[start] = nums[end];
            nums[end] = tmp;

            start++;
            end--;
        }
    }
};
```

# 4.除自身以外数组的乘积

[238. 除自身以外数组的乘积 - 力扣（LeetCode）](https://leetcode.cn/problems/product-of-array-except-self/description/?envType=study-plan-v2\&envId=top-100-liked "238. 除自身以外数组的乘积 - 力扣（LeetCode）")

```bash
给你一个整数数组 nums，返回 数组 answer ，其中 answer[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积 。

题目数据 保证 数组 nums之中任意元素的全部前缀元素和后缀的乘积都在  32 位 整数范围内。

请 不要使用除法，且在 O(n) 时间复杂度内完成此题。

 

示例 1:

输入: nums = [1,2,3,4]
输出: [24,12,8,6]
```

利用索引左侧所有数字的乘积和右侧所有数字的乘积（即前缀与后缀）相乘得到答案。

对于给定索引 i，将使用它左边所有数字的乘积乘以右边所有数字的乘积

1.  初始化 answer 数组，对于给定索引 i，answer\[i] 代表的是 i 左侧所有数字的乘积。
2.  试图节省空间，先把 answer 作为方法一的 L 数组。
3.  这种方法的唯一变化就是没有构造 R 数组。而是用一个遍历来跟踪右边元素的乘积。并更新数组 answer\[i]=answer\[i]∗R *。然后 R 更新为 R=R*∗nums\[i]，其中变量 R 表示的就是索引右侧数字的乘积。

```c++
class Solution {
public:
    // 原数组：       [1       2       3       4]
    // 左部分的乘积：   1       1      1*2    1*2*3
    // 右部分的乘积： 2*3*4    3*4      4      1
    // 结果：        1*2*3*4  1*3*4   1*2*4  1*2*3*1
    vector<int> productExceptSelf(vector<int>& nums) {
        int len = nums.size();
        if (len == 0) {
            return {};
        }

        std::vector<int> ans(len);
        ans[0] = 1;
        int tmp = 1;
        // 左侧部分乘积存在ans中
        for (int i = 1; i < len; i++) {
            ans[i] = ans[i - 1] * nums[i - 1];
        }
        // 右侧部分乘积，计算结果
        for (int i = len - 2; i >= 0; i--) {
            tmp *= nums[i + 1];
            ans[i] *= tmp;
        }

        return ans;
    }
};
```

# 5.缺失的第一个正数

[41. 缺失的第一个正数 - 力扣（LeetCode）](https://leetcode.cn/problems/first-missing-positive/description/?envType=study-plan-v2\&envId=top-100-liked "41. 缺失的第一个正数 - 力扣（LeetCode）")

```bash
给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。

请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。
 

示例 1：

输入：nums = [1,2,0]
输出：3
```

我们对数组进行遍历，对于遍历到的数 x，如果它在`  [1,N]  `的范围内，那么就将数组中的第 `x−1` 个位置（注意：数组下标从 0 开始）打上「标记」。在遍历结束之后，如果所有的位置都被打上了标记，那么答案是 `N+1`，否则答案是最小的没有打上标记的位置加 1。

-   将数组中所有小于等于 0 的数修改为 `N+1`；
-   遍历数组中的每一个数 x，它可能已经被打了标记，因此原本对应的数为 |x|，其中 || 为绝对值符号。如果$  |x| \in [1, N] $，那么给数组中的第$  |x| - 1 $个位置的数添加一个负号。注意如果它已经有负号，不需要重复添加；
-   在遍历完成之后，如果数组中的每一个数都是负数，那么答案是 N+1，否则答案是第一个正数的位置加 1。

```c++
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n = nums.size();
        // 将小于等于0的数字改为 n+1
        for (int& num : nums) {
            if (num <= 0) {
                num = n + 1;
            }
        }

        // 遍历数组，就将数组中的第 x−1 个位置（注意：数组下标从 0 开始）打上「标记」
        for (int i = 0; i < n; i++) {
            int num = abs(nums[i]);
            if (num <= n) {
                nums[num - 1] = -abs(nums[num - 1]);
            }
        }
        
        // 如果数组中的每一个数都是负数，那么答案是 N+1，否则答案是第一个正数的位置加 1。
        for (int i = 0; i < n; i++) {
            if (nums[i] > 0) {
                return i + 1;
            }
        }

        return n + 1;
    }
```
