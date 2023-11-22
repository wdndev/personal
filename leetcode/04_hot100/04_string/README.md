# 04子串

# 1.和为k的子数组

[560. 和为 K 的子数组 - 力扣（LeetCode）](https://leetcode.cn/problems/subarray-sum-equals-k/description/?envType=study-plan-v2\&envId=top-100-liked "560. 和为 K 的子数组 - 力扣（LeetCode）")

```bash
给你一个整数数组 nums 和一个整数 k ，请你统计并返回 该数组中和为 k 的子数组的个数 。

子数组是数组中元素的连续非空序列。


示例 1：

输入：nums = [1,1,1], k = 2
输出：2
```

1.  暴力枚举：可以枚举 \[0..i 里所有的下标 j 来判断是否符合条件，
2.  前缀和 + 哈希表优化

使用前缀和的方法可以解决这个问题，因为需要找到和为k的连续子数组的个数。通过计算前缀和，**可以将问题转化为求解两个前缀和之差等于k的情况**。

假设数组的前缀和数组为prefixSum，其中prefixSum\[i]表示从数组起始位置到第i个位置的元素之和。那么对于任意的两个下标i和j（i < j），如果prefixSum\[j] - prefixSum\[i] = k，即从第i个位置到第j个位置的元素之和等于k，那么说明从第i+1个位置到第j个位置的连续子数组的和为k。

**通过遍历数组，计算每个位置的前缀和，并使用一个哈希表来存储每个前缀和出现的次数**。在遍历的过程中，我们检查是否存在prefixSum\[j] - k的前缀和，如果存在，说明从某个位置到当前位置的连续子数组的和为k，我们将对应的次数累加到结果中。

这样，通过遍历一次数组，我们可以统计出和为k的连续子数组的个数，并且时间复杂度为O(n)，其中n为数组的长度。

```c++
class Solution {
public:
    // 1.暴力枚举
    int subarraySum1(vector<int>& nums, int k) {
        int count = 0;

        for (int start = 0; start < nums.size(); start++) {
            int sum = 0;
            for (int end = start; end >= 0; end--) {
                sum += nums[end];
                if (sum == k) {
                    count++;
                }
            }
        }

        return count;
    }

    // 2.前缀和 + 哈希表优化
    int subarraySum(vector<int>& nums, int k) {
        int count = 0;
        // prefixSum[i]表示从数组起始位置到第i个位置的元素之和
        int prefix_sum = 0;
        // hash表中存储前缀和出现的次数
        std::unordered_map<int, int> mp;
        // 初始化前缀和为0的次数为1
        mp[0] = 1;

        for (auto& x : nums) {
            prefix_sum += x;
            // 检查是否存在prefixSum[j] - k的前缀和，
            // 如果存在，说明从某个位置到当前位置的连续子数组的和为k
            if (mp.find(prefix_sum - k) != mp.end()) {
                count += mp[prefix_sum - k];
            }
            mp[prefix_sum]++;
        }

        return count;
    }
};
```

# 2.滑动窗口的最大值

[239. 滑动窗口最大值 - 力扣（LeetCode）](https://leetcode.cn/problems/sliding-window-maximum/description/?envType=study-plan-v2\&envId=top-100-liked "239. 滑动窗口最大值 - 力扣（LeetCode）")

```bash
给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

返回 滑动窗口中的最大值 。
```

1.  暴力解法，遍历循环
2.  队列

初始时，将数组 nums 的前 k 个元素放入优先队列中。每当向右移动窗口时，就可以把一个新的元素放入优先队列中，此时堆顶的元素就是堆中所有元素的最大值。然而这个最大值可能并不在滑动窗口中，在这种情况下，这个值在数组 nums 中的位置出现在滑动窗口左边界的左侧。因此，当后续继续向右移动窗口时，这个值就永远不可能出现在滑动窗口中了，可以将其永久地从优先队列中移除。

不断地移除堆顶的元素，直到其确实出现在滑动窗口中。此时，堆顶元素就是滑动窗口中的最大值。为了方便判断堆顶元素与滑动窗口的位置关系，我们可以在优先队列中存储二元组 (num,index)，表示元素 num 在数组中的下标为 index。

```c++
// 1.暴力 O(n*k)
// 2.deque O(n)
class Solution {
public:
    // 1.暴力, 超时
    vector<int> maxSlidingWindow1(vector<int>& nums, int k) {
        std::vector<int> ans;
        if (nums.size() != 0) {
            int tmp_max = this->get_vector_max(nums, 0, k);
            for (int i = 0; i < nums.size() - (k - 1); i++) {
                tmp_max = this->get_vector_max(nums, i, i + k);
                ans.push_back(tmp_max);
            }
        }

        return ans;
    }

    // 2.队列
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int n = nums.size();
        std::deque<int> que;

        // 将前k个元素的下标加入队列中，其中队列头为最大值，队列尾为最小值
        for (int i = 0; i < k; i++) {
            // 将小于队列低的元素加在后面
            while (!que.empty() && nums[i] >= nums[que.back()]) {
                que.pop_back();
            }

            que.push_back(i);
        }

        // 将前k个元素的最大值加进去
        std::vector<int> ans = {nums[que.front()]};
        // 开始遍历
        for (int i = k; i < n; i++) {
            while (!que.empty() && nums[i] >= nums[que.back()]) {
                que.pop_back();
            }

            que.push_back(i);
            //
            while (que.front() <= i - k) {
                que.pop_front();
            }
            ans.push_back(nums[que.front()]);
        }

        return ans;        
    }

private:
    int get_vector_max(std::vector<int>& nums, int start, int end) {
        int max_val = INT_MIN;
        for (int i = start; i <= end; i++) {
            max_val = std::max(max_val, nums[i]);
        }

        return max_val;
    }
};
```

# 3.最小覆盖子串

[76. 最小覆盖子串 - 力扣（LeetCode）](https://leetcode.cn/problems/minimum-window-substring/description/?envType=study-plan-v2\&envId=top-100-liked "76. 最小覆盖子串 - 力扣（LeetCode）")

```bash
给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。
```

滑动窗口思想来解决，使用两个指针，right和left；right指针用来扩展滑动窗口，left指针用来收缩滑动窗口；在任意时刻，只有一个指针运动，而另一个指针保持静止。

**在s上滑动窗口，通过移动right指针不断扩展窗口，当滑动窗口中包含所有字符后，判断是否可以收缩，移动left**

使用哈希表表示字符串t中所有的字符及个数，

用一个动态哈希表维护滑动窗口中所有的字符及个数

如果这个动态哈希表中的所有自读，且对应的个数小于t的哈希表各个字符的个数，那么当前滑动窗口是可行的

```c++
class Solution {
public:
    // 滑动窗口思想来解决，使用两个指针，right和left
    // right指针用来扩展滑动窗口，left指针用来收缩滑动窗口
    // 在任意时刻，只有一个指针运动，而另一个指针保持静止
    // 在s上滑动窗口，通过移动right指针不断扩展窗口，当滑动窗口中包含所有字符后，判断是否可以收缩，移动left
    // 使用哈希表表示字符串t中所有的字符及个数，
    // 用一个动态哈希表维护滑动窗口中所有的字符及个数
    // 如果这个动态哈希表中的所有自读，且对应的个数小于t的哈希表各个字符的个数，那么当前滑动窗口是可行的
    string minWindow(string s, string t) {
        // 将t中的字符，全部加入t hash表中
        for (auto& c : t) {
            ori_count[c]++;
        }

        // 滑动窗口左右定义
        int left = 0;
        int right = -1;

        // 结果字符串长度，及索引
        int ret_str_len = s.size() + 1;
        int ret_left = -1;
        // int ret_right = -1;

        // 如果滑动窗口右侧没有到s字符串的末尾，扩展
        while (right < int(s.size())) {
            // 判断滑动窗口右侧right移动后的字符是否在t的hash表中
            // 如果在，加入滑动窗口hash表
            if (ori_count.find(s[++right]) != ori_count.end() ) {
                dy_count[s[right]]++;
            }

            // 检查t中的字符，是否全部包含进滑动窗口的hash表中？
            // 如果包含，开始缩减滑动窗口大小
            // 否则，继续扩展滑动窗口
            while (check() && left <= right) {
                // 更新结果字符串的长度和坐标
                if (right - left + 1 < ret_str_len) {
                    ret_str_len = right - left + 1;
                    ret_left = left;
                }
                // 如果最左侧left字符不在t hash表中，则缩减滑动窗口
                if (ori_count.find(s[left]) != ori_count.end() ) {
                    dy_count[s[left]] --;
                }
                left++;
            }

        }

        return ret_left == -1 ? std::string() : s.substr(ret_left, ret_str_len);
    }
private:
    // t哈希表
    std::unordered_map<char, int> ori_count;
    // 滑动窗口hash表
    std::unordered_map<char, int> dy_count;

    // 判断t中的字符是否全包包含在滑动窗口的hash表中
    bool check() {
        for (auto& p : ori_count) {
            if (dy_count[p.first] < p.second) {
                return false;
            }
        }

        return true;
    }
};
```
