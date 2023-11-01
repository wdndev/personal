/*
 * @lc app=leetcode.cn id=912 lang=cpp
 *
 * [912] 排序数组
 *
 * https://leetcode.cn/problems/sort-an-array/description/
 *
 * algorithms
 * Medium (50.48%)
 * Likes:    930
 * Dislikes: 0
 * Total Accepted:    585.7K
 * Total Submissions: 1.2M
 * Testcase Example:  '[5,2,3,1]'
 *
 * 给你一个整数数组 nums，请你将该数组升序排列。
 * 
 * 
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [5,2,3,1]
 * 输出：[1,2,3,5]
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [5,1,1,2,0,0]
 * 输出：[0,0,1,1,2,5]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= nums.length <= 5 * 10^4
 * -5 * 10^4 <= nums[i] <= 5 * 10^4
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    vector<int> sortArray(vector<int>& nums) {
        if (nums.size() == 0) {
            return {};
        }
        // // 1. 插入排序
        // this->select_sort(nums);

        // // // 2.1 直接插入排序
        // this->inert_sort(nums);

        // // 2.2 折半插入排序
        // this->bin_insert_sort(nums);

        // // 2.3 希尔排序
        // this->shell_sort(nums);

        // 2.4 冒泡
        // this->bulle_sort(nums);

        // 3.1 快速排序
        // this->quick_sort(nums, 0, nums.size() - 1);

        // 3.2 归并排序
        // this->merge_sort(nums, 0, nums.size() - 1);

        // 3.3 堆排序
        this->heap_sort(nums);

        return nums;
    }
private:
    // 1. 选择排序 O(n^2)  超出时间限制
    // 每一趟在待排序元素中选取关键字最小的元素加入有序子序列中
    void select_sort(std::vector<int>& nums) {
        int len = nums.size();
        // 记录最小元素位置
        int min_idx = -1;
        // 遍历
        for (int i = 0; i < len; i++) {
            min_idx = i;
            // 选择最小元素
            for (int j = i + 1; j < len; j++) {
                if (nums[j] < nums[min_idx]) {
                    min_idx = j;
                }
            }
            // 交换元素
            if (min_idx != i) {
                int tmp = nums[min_idx];
                nums[min_idx] = nums[i];
                nums[i] = tmp;
            }
        }
    }

    // 2.1 直接插入排序 超时
    // 将0号位置放哨兵：要插入的元素
    // 每次将一个待排序的记录按其关键字插入到前面已排号的序列中
    void inert_sort(std::vector<int>& nums) {
        int len = nums.size();
        if (len <= 1) {
            return;
        }
        // 依次将 2~n 插入到前面已排序的序列
        for (int i = 1; i < len; i++) {
            // i 小于前驱，将 i 插入有序表
            if (nums[i] < nums[i - 1]) {
                int tmp = nums[i];
                int j = i - 1;
                // 从后往前查找待插入得位置
                for (; j >=0 && tmp < nums[j]; j--) {
                    // 向后移动元素
                    nums[j + 1] = nums[j];
                }
                // 插入数据
                nums[j + 1] = tmp;
            }
        }
    }

    // 2.2 折半插入排序 超时
    void bin_insert_sort(std::vector<int>& nums) {
        int left, mid, right;
        for (int i = 1; i < nums.size(); i++) {
            int tmp = nums[i];

            left = 0;
            right = i - 1;
            // 查找
            while (left <= right) {
                mid = left + (right - left) / 2;
                // 查找左半部分
                if (nums[mid] > tmp) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }

            // 后移元素，空出插入位置
            // 注意范围，left指针指向选中的元素
            for (int j = i - 1; j >= left; j--) {
                nums[j + 1] = nums[j];
            }
            // 插入
            nums[left] = tmp;
        }
    }

    // 2.3 希尔排序 超时
    // 先将排序表分割成若干如L[i, i+d, i+2d, ..., i+kd] 的“特殊”子表，
    // 对各个子表进行直接插入排序。
    // 缩小增量d，重复上述过程，直到d=1为止。
    void shell_sort(std::vector<int>& nums) {
        int len = nums.size();
        // 步长
        for (int dk = len / 2; dk >= 1; dk = dk / 2) {
            for (int i = dk + 1; i < len; i++) {
                if (nums[i] < nums[i - dk]) {
                    // 将i插入有序增量子表
                    int tmp = nums[i];
                    // 元素后移
                    int j = i - dk;
                    for (; i >= 0 && tmp < nums[j]; j -= dk) {
                        nums[j + dk] = nums[j];
                    }
                    // 插入
                    nums[j + dk] = tmp;
                }
            }
        }
    }

    // 2.4 冒泡排序 超时
    // 从后往前（或从前往后）两两比较相邻元素的值，若为逆序，
    // 则交换它们，直到序列比较完。称这个过程为“一趟”冒泡排序，
    // 最多只需n-1趟排序

    // 每一趟排序后都可以使一个元素的移动到最终位置，
    // 以确定最终位置的元素在之后的处理中无需对比

    // 如果某一趟排序过程中未发生“交换”，则算法可以提前结束
    void bulle_sort(std::vector<int>& nums) {
        int len = nums.size();
        if (len <= 1) {
            return;
        }
        // 提前退出标志位
        bool flag = false;
        for (int i = 0; i < len; i++)
        {
            flag = false;
            for (int j = 0; j < len - i - 1; j++)
            {
                // 前面的元素比后面的大，交换顺序
                if (nums[j] > nums[j + 1])
                {
                    int tmp = nums[j];
                    nums[j] = nums[j + 1];
                    nums[j + 1] = tmp;
                    // 数据交换标志位
                    flag = true;
                }
            }
            // 若没有数据交换，提前退出
            if (!flag)
                break;
        }
    }

    // 3.1 快速排序
    // 一次划分区间
    int partite_region(std::vector<int>& nums, int left, int right)
    {
        // 当前表的第一个元素作为枢轴，对其划分
        int pivot = nums[left];
        // 循环条件
        while (left < right)
        {
            // high向前寻找比枢轴点小的元素
            while (left < right && nums[right] > pivot)
                right--;
            // 将此小的元素移动到枢轴点左端
            nums[left] = nums[right];
            // low向后寻找比枢轴点大的元素
            while (left < right && nums[left] <= pivot)
                left++;
            // 将此大的元素移动到枢轴点右端
            nums[right] = nums[left];
        }
        // 枢轴点元素放到最终位置
        nums[left] = pivot;
        // 放回枢轴点元素
        return left;
    }
    void quick_sort(std::vector<int>& nums, int left, int right)
    {
        if (left < right)
        {
            // 划分区间
            int pivot_pos = partite_region(nums, left, right);
            // 一次对两个子表进行递归排序
            this->quick_sort(nums, left, pivot_pos - 1);
            this->quick_sort(nums, pivot_pos + 1, right);
        }
    }

    // 3.1 归并排序
    // 先排序左右子数组，然后合并两个有序子数组
    // 1. 把长度为n的输入序列分成两个长度为n/2的子序列
    // 2. 对这两个子序列分别采用归并排序:
    // 3. 将两个排序好的子序列合并成一个最终的排序序列
    void merge(std::vector<int>& nums, int left, int mid, int right)
    {
        // 辅助数组
        std::vector<int> tmp(right - left + 1);
        // // 表nums中的元素，全部复制到tmp中
        // for (int k = left; k <= right; k++)
        //     tmp[k] = nums[k];
        int left_idx = left;
        int right_idx = mid + 1;
        int tmp_index = 0;
        // 比较tmp的左右两段中的元素,将较小值复制到L中
        while (left_idx <= mid && right_idx <= right)
        {
            // 两个元素相等时，优先使用靠前的那个（稳定性）
            if (nums[left_idx] <= nums[right_idx])
                tmp[tmp_index++] = nums[left_idx++];
            else
                tmp[tmp_index++] = nums[right_idx++];
        }

        // 若第一个表未检测完，复制
        while(left_idx <= mid)
            tmp[tmp_index++] = nums[left_idx++];
        // 若第二个表未检测完，复制
        while (right_idx <= right)
            tmp[tmp_index++] = nums[right_idx++];

        // 复制tmp到nums
        for (int k = 0; k < tmp.size(); k++) {
            nums[left + k] = tmp[k];
        }

    }
    void merge_sort(std::vector<int>& nums, int left, int right)
    {
        if (left < right)
        {
            int mid = left + (right - left) / 2;
            this->merge_sort(nums, left, mid);
            this->merge_sort(nums, mid + 1, right);
            this->merge(nums, left, mid, right);
        }
    }


    // 3.3 堆排序
    // 从下往上堆化
    void heapify(std::vector<int>& nums, int n, int k) {
        //std::cout << "2222";
        while (true)
        {
            // 寻找父结点的两个子结点中最大的一个
            int max_pos = k;
            if (k * 2 < n && nums[k] < nums[k * 2])
                max_pos = k * 2;
            if (k * 2 + 1 < n && nums[max_pos] < nums[k * 2 + 1])
                max_pos = k * 2 + 1;
            // 若子结点比自己都小，父结点就是最大的
            if (max_pos == k)
                break;
            // 交换元素
            int tmp_elem = nums[k];
            nums[k] = nums[max_pos];
            nums[max_pos] = tmp_elem;
            // 更新
            k = max_pos;
        }
    }

    void heap_sort(std::vector<int>& nums) {
        int len = nums.size();
        if (len == 0) {
            return;
        }

        // 建堆
        for (int i = len / 2 - 1; i >= 0; i--) {
            heapify(nums,len, i);
        }

        // 排序
        for (int i = len - 1; i >= 0; i--) {
            int tmp = nums[i];
            nums[i] = nums[0];
            nums[0] = tmp;

            heapify(nums, i, 0);
        }
    }

};
// @lc code=end

