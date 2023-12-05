# 17.技巧

# 1.只出现一次的数字

[136. 只出现一次的数字 - 力扣（LeetCode）](https://leetcode.cn/problems/single-number/description/?envType=study-plan-v2\&envId=top-100-liked "136. 只出现一次的数字 - 力扣（LeetCode）")

```json
给你一个 非空 整数数组 nums ，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。

你必须设计并实现线性时间复杂度的算法来解决此问题，且该算法只使用常量额外空间。


示例 1 ：

输入：nums = [2,2,1]
输出：1
```

如果不考虑时间复杂度和空间复杂度的限制，这道题有很多种解法，可能的解法有如下几种。

-   **使用集合存储数字**。遍历数组中的每个数字，如果集合中没有该数字，则将该数字加入集合，如果集合中已经有该数字，则将该数字从集合中删除，最后剩下的数字就是只出现一次的数字。
-   **使用哈希表存储每个数字和该数字出现的次数**。遍历数组即可得到每个数字出现的次数，并更新哈希表，最后遍历哈希表，得到只出现一次的数字。
-   **使用集合存储数组中出现的所有数字，并计算数组中的元素之和**。由于集合保证元素无重复，因此计算集合中的所有元素之和的两倍，即为每个元素出现两次的情况下的元素之和。由于数组中只有一个元素出现一次，其余元素都出现两次，因此用集合中的元素之和的两倍减去数组中的元素之和，剩下的数就是数组中只出现一次的数字。

上述三种解法都需要额外使用 O(n) 的空间，其中 n 是数组长度。

答案是使用位运算。对于这道题，可使用异或运算 $\oplus$。异或运算有以下性质。

-   `x^0 = x`
-   `x^1s = ~x`  (注意 1s = \~0， “全1”)
-   `x^(~x) = 1s`
-   `x^x = 0`
-   `c = a ^ b → a^c = b, b^c = a` (交换两个数)
-   `a^b^c = a^(b^c)=(a^b)^c` (associative)

```c++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int ans = 0;
        for (auto n : nums)
            ans ^= n;
        
        return ans;
    }
};
```

# 2.多数元素

[169. 多数元素 - 力扣（LeetCode）](https://leetcode.cn/problems/majority-element/description/?envType=study-plan-v2\&envId=top-100-liked "169. 多数元素 - 力扣（LeetCode）")

```json
给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

 
示例 1：

输入：nums = [3,2,3]
输出：3
```

1.  哈希表，统计元素次数
2.  排序，去中间那个元素

```c++
class Solution {
public:
    // 1.哈希表，统计元素次数
    int majorityElement1(vector<int>& nums) {
        std::unordered_map<int, int> counts;
        int max_cnt = 0;
        int max_elem = 0;
        for (auto& num : nums) {
            counts[num]++;
            if (counts[num] > max_cnt) {
                max_elem = num;
                max_cnt = counts[num];
            }
        }

        return max_elem;
    }

    // 2.排序，取中间元素
    int majorityElement(vector<int>& nums) {
        std::sort(nums.begin(), nums.end());
        return nums[nums.size() / 2];
    }
};
```

# 3.颜色分类

[75. 颜色分类 - 力扣（LeetCode）](https://leetcode.cn/problems/sort-colors/description/?envType=study-plan-v2\&envId=top-100-liked "75. 颜色分类 - 力扣（LeetCode）")

```json
给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。

我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

必须在不使用库内置的 sort 函数的情况下解决这个问题。

 

示例 1：

输入：nums = [2,0,2,1,1,0]
输出：[0,0,1,1,2,2]
```

1.  统计012数字个数，重新排列
2.  单指针：先将数组中所有0交换到头部，再将1交换到0之后
3.  双指针：左右指针移动，交换0和2

```c++
class Solution {
public:
    // 2.双指针
    void sortColors(vector<int>& nums) {
        // 左右指针
        int left = 0;
        int right = nums.size() - 1;
        
        // 遍历所有数组
        for (int i = 0; i < nums.size(); i++) {
            // 先处理 2，移动到最后
            while (i <= right && nums[i] == 2) {
                swap(nums[i], nums[right]);
                right--;
            }

            // 再处理1
            if (nums[i] == 0) {
                swap(nums[i], nums[left]);
                left++;
            }
        }
    }
};
```

# 4.下一个排列

[31. 下一个排列 - 力扣（LeetCode）](https://leetcode.cn/problems/next-permutation/description/?envType=study-plan-v2\&envId=top-100-liked "31. 下一个排列 - 力扣（LeetCode）")

```python
整数数组的一个 排列  就是将其所有成员以序列或线性顺序排列。

- 例如，arr = [1,2,3] ，以下这些都可以视作 arr 的排列：[1,2,3]、[1,3,2]、[3,1,2]、[2,3,1] 。

整数数组的 下一个排列 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，那么数组的 下一个排列 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。

- 例如，arr = [1,2,3] 的下一个排列是 [1,3,2] 。
- 类似地，arr = [2,3,1] 的下一个排列是 [3,1,2] 。
- 而 arr = [3,2,1] 的下一个排列是 [1,2,3] ，因为 [3,2,1] 不存在一个字典序更大的排列。
给你一个整数数组 nums ，找出 nums 的下一个排列。

必须 原地 修改，只允许使用额外常数空间。
```

**算法推导**

-   希望下一个数 比当前数大，这样才满足 “下一个排列” 的定义。因此只需要 **将后面的「大数」与前面的「小数」交换**，就能得到一个更大的数。比如 123456，将 5 和 6 交换就能得到一个更大的数 123465。
-   还希望下一个数 增加的幅度尽可能的小，这样才满足“下一个排列与当前排列紧邻“的要求。为了满足这个要求，需要：
    1.  在\*\* 尽可能靠右的低位 进行交换\*\*，需要 从后向前 查找
    2.  将一个 **尽可能小的「大数」 与前面的「小数」交换**。比如 123465，下一个排列应该把 5 和 4 交换而不是把 6 和 4 交换
    3.  将「大数」换到前面后，需要将「大数」后面的所有数 重置为升序，升序排列就是最小的排列。以 123465 为例：首先按照上一步，交换 5 和 4，得到 123564；然后需要将 5 之后的数重置为升序，得到 123546。显然 123546 比 123564 更小，123546 就是 123465 的下一个排列

**算法过程**

标准的 “下一个排列” 算法可以描述为：

1.  **从后向前** 查找第一个 \*\*相邻升序 \*\*的元素对 `(i,j)`，满足 `A[i] < A[j]`。此时 `[j,end) `必然是降序
2.  在 `[j,end)` 从后向前 查找第一个满足`  A[i] < A[k]  `的 `k`。`A[i]`、`A[k]` 分别就是上文所说的「小数」、「大数」
3.  将` A[i]` 与 `A[k]` 交换
4.  可以断定这时 `[j,end)` 必然是降序，逆置 `[j,end)`，使其升序
5.  如果在步骤 1 找不到符合的相邻元素对，说明当前 `[begin,end) `为一个降序顺序，则直接跳到步骤 4

该方法支持数据重复，且在 C++ STL 中被采用。

```c++
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int l=nums.size(),i=l-2;
        while(i>=0&&nums[i]>=nums[i+1]){
            i--;
        }
        if(i>=0){
            int j=l-1;
            //找出nums[i]后面大于nums[i]的最小数的下标
            while(nums[j]<=nums[i]){
                j--;
            }
            swap(nums[i],nums[j]);
            reverse(nums.begin()+i+1,nums.end());//交换完后对nums[i]后面的数字进行从小到大排列
            //因为此时nums.begin()+i+1到nums.end()一定是降序排列,所以只需reverse就是从小到大排列了
        }else{//说明是最大排列,下一个应该是最小排列
            reverse(nums.begin(),nums.end());
        }
        return;
    }
};
```

# 5.寻找重复数

[287. 寻找重复数 - 力扣（LeetCode）](https://leetcode.cn/problems/find-the-duplicate-number/description/?envType=study-plan-v2\&envId=top-100-liked "287. 寻找重复数 - 力扣（LeetCode）")

```bash
给定一个包含 n + 1 个整数的数组 nums ，其数字都在 [1, n] 范围内（包括 1 和 n），可知至少存在一个重复的整数。

假设 nums 只有 一个重复的整数 ，返回 这个重复的数 。

你设计的解决方案必须 不修改 数组 nums 且只用常量级 O(1) 的额外空间。

 

示例 1：

输入：nums = [1,3,4,2,2]
输出：2
示例 2：

输入：nums = [3,1,3,4,2]
输出：3
```

二分查找

定义 `cnt[i]`表示 `nums`数组中小于等于 i 的数有多少个，假设我们重复的数是 `target`，那么 `[1,target−1]`里的所有数满足 `cnt[i]≤i`，`[target,n]` 里的所有数满足 `cnt[i]>i`，具有单调性。

如果知道 `cnt[]` 数组随数字 i 逐渐增大具有单调性（即 target 前 `cnt[i]≤i`，target 后 `cnt[i]>i`），那么我们就可以直接利用二分查找来找到重复的数。

```c++
class Solution {
public:
    // 1.哈希表，不符合题目要求
    int findDuplicate1(vector<int>& nums) {
        std::unordered_map<int, int> count;
        for (const auto& n : nums) {
            count[n]++;
            if (count[n] > 1) {
                return n;
            }
        }
        return -1;
    }

    // 2.二分查找
    int findDuplicate(vector<int>& nums) {
        int n = nums.size();
        int left = 1;
        int right = n - 1;
        int ans = -1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            int cnt = 0;
            for (int i = 0; i < n; i++) {
                cnt += nums[i] <= mid;
            }

            if (cnt <= mid) {
                left = mid + 1;
            } else {
                right = mid - 1;
                ans = mid;
            }
        }

        return ans;
    }
};
```
