# 11 bin\_search

# 1.搜索插入位置

[35. 搜索插入位置 - 力扣（LeetCode）](https://leetcode.cn/problems/search-insert-position/description/?envType=study-plan-v2\&envId=top-100-liked "35. 搜索插入位置 - 力扣（LeetCode）")

```python
给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

请必须使用时间复杂度为 O(log n) 的算法。

示例 1:

输入: nums = [1,3,5,6], target = 5
输出: 2
```

```c++
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int left = 0;
        int right = nums.size() - 1;
        int tgt_idx = 0;

        while (left <= right) {
            // 
            int mid = left + (right - left) / 2;

            if (nums[mid] == target) {
                tgt_idx = mid;
                break;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
                // 注意，是左侧
                tgt_idx = left;
            }
        }
        return tgt_idx;
    }
};
```

# 2.搜索二维矩阵

[74. 搜索二维矩阵 - 力扣（LeetCode）](https://leetcode.cn/problems/search-a-2d-matrix/description/?envType=study-plan-v2\&envId=top-100-liked "74. 搜索二维矩阵 - 力扣（LeetCode）")

```python
给你一个满足下述两条属性的 m x n 整数矩阵：

每行中的整数从左到右按非严格递增顺序排列。
每行的第一个整数大于前一行的最后一个整数。
给你一个整数 target ，如果 target 在矩阵中，返回 true ；否则，返回 false 。

示例 1：
输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
输出：true

```

1.  将二维矩阵当作一维矩阵来出来，因为是严格递增的

```c++
class Solution {
public:
    // 1.将二维矩阵展平为一维，进行二分查找
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int row = matrix.size();
        int col = matrix[0].size();
        
        int left = 0;
        int right = row * col - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            
            if (matrix[mid / col][mid % col] > target) {
                right = mid - 1;
            } else if (matrix[mid / col][mid % col] < target) {
                left = mid + 1;
            } else {
                return true;
            }
        }

        return false;
    }
};
```

# 3.在排序数组中查找元素的第一个和最后一个位置

[34. 在排序数组中查找元素的第一个和最后一个位置 - 力扣（LeetCode）](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/description/?envType=study-plan-v2\&envId=top-100-liked "34. 在排序数组中查找元素的第一个和最后一个位置 - 力扣（LeetCode）")

```python
给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 target，返回 [-1, -1]。

你必须设计并实现时间复杂度为 O(log n) 的算法解决此问题。

 

示例 1：

输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]
示例 2：

输入：nums = [5,7,7,8,8,10], target = 6
输出：[-1,-1]
```

1.  两次二分查找，一次查找第一个位置，第二次查找第二个位置
2.  两次二分查找，找target加减0.5的数

```c++
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int start_idx = this->bsearch_start(nums, target);
        int end_idx = this->bsearch_end(nums, target);


        return {start_idx, end_idx};
    }

    int bsearch_start(const std::vector<int>& nums, int target) {
        int left = 0;
        int right = nums.size() - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                // 注意
                if ((mid == 0) || (nums[mid - 1] != target)) {
                    return mid;
                } else {
                    right = mid - 1;
                }

            }
        }

        return -1;
    }

    int bsearch_end(const std::vector<int>& nums, int target) {
        int left = 0;
        int right = nums.size() - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                // 注意
                if ((mid == nums.size() - 1) || (nums[mid + 1] != target)) {
                    return mid;
                } else {
                    left = mid + 1;
                }
            }
        }

        return -1;
    }
};
```

```c++
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {

        // 两次二分
        int start = this->bsearch(nums, target-0.5);
        int end = this->bsearch(nums, target+0.5);
        if (start == end) {  // 找不到
            return {-1, -1};
        }

        return {start, end -1};
    }

    int bsearch(std::vector<int>& nums, float target){
        int left = 0;
        int right = nums.size() - 1;

        while(left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left;
    }
};
```

# 4.搜索旋转排序数组

[33. 搜索旋转排序数组 - 力扣（LeetCode）](https://leetcode.cn/problems/search-in-rotated-sorted-array/description/?envType=study-plan-v2\&envId=top-100-liked "33. 搜索旋转排序数组 - 力扣（LeetCode）")

```python
整数数组 nums 按升序排列，数组中的值 互不相同 。

在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。

给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。

你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。


示例 1：

输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4
```

使用二分查找：

-   如果`target`在`[mid+1, high]`序列中，则`low=mid+1`，否则，`high=mid`，关键是如何判断`target`在`[mid+1, high]`序列中，具体判断如下：
-   当`[0, mid]`序列是升序：`nums[0] ≤ nums[mid]`，当t`arget>nums[mid] || target <nums[0]`，则向后规约；
-   当`[0, mid]`序列存在旋转位：`nums[0] > nums[mid]`，当`target<nums[0] && target >nums[mid]`，则向后规约；
-   其他情况就是向前规约了

循环判断，直到排除到只剩一个元素时，退出循环，如果该元素和target相同，直接返回下标，否则返回-1.

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {

        int left = 0;
        int right = nums.size() - 1;
        int ans = -1;
        while (left < right) {
            // 注意此处，主要是为了防止数据溢出
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                ans = mid;
            } 
            // 当[0,mid]有序时，向后规约条件
            if (nums[0] <= nums[mid] && (target > nums[mid] || target < nums[0])) {
                left = mid + 1;
            
            // 当[0, mid]发生旋转时，向后规约条件
            } else if (target > nums[mid] && target < nums[0]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        return left == right && nums[left] == target ? left : -1;
    }
};
```

# 5.搜索旋转排序数组中的最小值

[153. 寻找旋转排序数组中的最小值 - 力扣（LeetCode）](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/description/?envType=study-plan-v2\&envId=top-100-liked "153. 寻找旋转排序数组中的最小值 - 力扣（LeetCode）")

```python
已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,2,4,5,6,7] 在变化后可能得到：
- 若旋转 4 次，则可以得到 [4,5,6,7,0,1,2]
- 若旋转 7 次，则可以得到 [0,1,2,4,5,6,7]
注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。

给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。

你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。


示例 1：

输入：nums = [3,4,5,1,2]
输出：1
解释：原数组为 [1,2,3,4,5] ，旋转 3 次得到输入数组。
```

1.  暴力
2.  二分查找

-   在二分查找的每一步中，左边界为 left，右边界为 right，区间的中点为 mid，最小值就在该区间内。
-   将中轴元素 `nums[mid`] 与右边界元素` nums[right]`进行比较，可能会有以下的三种情况：
-   第一种情况是 `nums[mid]<nums[right]`。这说明 `nums[mid] `是最小值右侧的元素，因此可以忽略二分查找区间的右半部分。
-   第二种情况是 `nums[mid]>nums[right]`。这说明 `nums[mid]`是最小值左侧的元素，因此可以忽略二分查找区间的左半部分。
-   由于数组不包含重复元素，并且只要当前的区间长度不为 1，mid 就不会与 right 重合；
-   而如果当前的区间长度为 1，这说明已经可以结束二分查找了。因此不会存在 `nums[mid]=nums[right]`的情况。
-   当二分查找结束时，我们就得到了最小值所在的位置。

```c++
class Solution {
public:
    // 1.暴力查找
    int findMin1(vector<int>& nums) {
        int min_num = INT_MAX;
        for (const auto& n : nums) {
            min_num = std::min(min_num, n);
        }

        return min_num;
    }

    // 2.二分查找
    // 在二分查找的每一步中，左边界为 left，右边界为 right，区间的中点为 mid，最小值就在该区间内。
    // 将中轴元素 nums[mid]与右边界元素 nums[right]进行比较，可能会有以下的三种情况：

    // 第一种情况是 nums[mid]<nums[right]。这说明 nums[mid] 是最小值右侧的元素，因此可以忽略二分查找区间的右半部分。
    // 第二种情况是 nums[mid]>nums[right]。这说明 nums[mid]是最小值左侧的元素，因此可以忽略二分查找区间的左半部分。
    // 由于数组不包含重复元素，并且只要当前的区间长度不为 1，mid 就不会与 right 重合；
    // 而如果当前的区间长度为 1，这说明已经可以结束二分查找了。因此不会存在 nums[mid]=nums[right]的情况。

    // 当二分查找结束时，我们就得到了最小值所在的位置。
    int findMin(vector<int>& nums) {
        int left = 0;
        int right = nums.size() - 1;
        int idx = -1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < nums[right]) {
                right = mid;
            } else if (nums[mid] > nums[right]){
                left = mid + 1;
            }
        }

        return nums[left];
    }
};
```

# 6.寻找两个正序数组的中位数

[4. 寻找两个正序数组的中位数 - 力扣（LeetCode）](https://leetcode.cn/problems/median-of-two-sorted-arrays/description/?envType=study-plan-v2\&envId=top-100-liked "4. 寻找两个正序数组的中位数 - 力扣（LeetCode）")

```python
给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。

算法的时间复杂度应该为 O(log (m+n)) 。

 

示例 1：

输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3] ，中位数 2
示例 2：

输入：nums1 = [1,2], nums2 = [3,4]
输出：2.50000
解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
```

1.  暴力，合并两个数组，然后找出中位数
2.  二分搜索变形：

用 len 表示合并后数组的长度，如果是奇数，我们需要知道第 `（len+1）/2 `个数就可以了，如果遍历的话需要遍历 `int(len/2 ) + 1` 次。如果是偶数，我们需要知道第` len/2`和 `len/2+1 `个数，也是需要遍历 `len/2+1` 次。所以遍历的话，奇数和偶数都是 `len/2+1` 次。

返回中位数的话，奇数需要最后一次遍历的结果就可以了，偶数需要最后一次和上一次遍历的结果。所以我们用两个变量 `left `和 `right`，`right`保存当前循环的结果，在每次循环前将 `right` 的值赋给 `left`。这样在最后一次循环的时候，`left` 将得到 `right` 的值，也就是上一次循环的结果，接下来 `right` 更新为最后一次的结果。

循环中该怎么写，什么时候 `nums1 `数组后移，什么时候 `nums2 `数组后移。用 `start_1`和 `start_2`分别表示当前指向 `nums1 `数组和 `nums2 `数组的位置。如果 `start_1`还没有到最后并且此时 `nums1`位置的数字小于 `nums2 `位置的数组，那么就可以后移了。也就是`start_1＜m&&nums1[start_1]< nums2[start_2]`。

但如果 `nums2 `数组此刻已经没有数字了，继续取数字` nums2[ start_2]`，则会越界，所以判断下 `start_2`是否大于数组长度了，这样 `||` 后边的就不会执行了，也就不会导致错误了，所以增加为 `start_1＜m&&(start_2) >= n||nums1[start_1]<nums2[start_2])` 。

```c++
class Solution {
public:
    // 1.暴力，合并两个数组
    double findMedianSortedArrays1(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size();
        int n = nums2.size();
        std::vector<int> new_nums(m + n);
        if (m == 0) {
            return this->get_mid_num(nums2);
        }
        if (n == 0) {
            return this->get_mid_num(nums1);
        }

        int idx = 0;
        int num1_idx = 0;
        int num2_idx = 0;
        while (idx != m + n) {
            if (num1_idx == m) {
                while (num2_idx < n) {
                     new_nums[idx++] = nums2[num2_idx++];
                }
                break;
            }

            if (num2_idx == n) {
                while (num1_idx < m) {
                     new_nums[idx++] = nums1[num1_idx++];
                }
                break;
            }

            if (nums1[num1_idx] < nums2[num2_idx]) {
                new_nums[idx++] = nums1[num1_idx++];
            } else {
                new_nums[idx++] = nums2[num2_idx++];
            }
        }

        return this->get_mid_num(new_nums);
    }

    // 2.二分查找变形
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size();
        int n = nums2.size();
        int len = m + n;

        // 上一次循环的结果
        int left = -1;
        // 本次循环的结果
        int right = -1;

        int start_1 = 0;
        int start_2 = 0;

        for (int i = 0; i <= len / 2; i++) {
            left = right;
            if (start_1 < m && (start_2 >= n || nums1[start_1] < nums2[start_2])) {
                right = nums1[start_1++];
            } else {
                right = nums2[start_2++];
            }
        }

        if (len % 2 == 0) {
            return (left + right) / 2.0;
        } else {
            return right;
        }
    }

private:
    double get_mid_num(std::vector<int>& nums) {
        int num_size = nums.size();
        if (num_size % 2 == 0) {
            return (nums[num_size / 2 - 1] + nums[num_size / 2]) / 2.0;
        } else {
            return nums[num_size / 2];
        }
    }
};
```
