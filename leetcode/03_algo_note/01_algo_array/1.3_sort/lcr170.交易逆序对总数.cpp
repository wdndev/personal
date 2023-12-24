// [LCR 170. 交易逆序对的总数 - 力扣（LeetCode）](https://leetcode.cn/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

// ```C++
// 在股票交易中，如果前一天的股价高于后一天的股价，则可以认为存在一个「交易逆序对」。请设计一个程序，输入一段时间内的股票交易记录 record，返回其中存在的「交易逆序对」总数。

 

// 示例 1:

// 输入：record = [9, 7, 5, 4, 6]
// 输出：8
// 解释：交易中的逆序对为 (9, 7), (9, 5), (9, 4), (9, 6), (7, 5), (7, 4), (7, 6), (5, 4)。
// ```


class Solution {
public:
    int reversePairs(vector<int>& record) {
        m_count = 0;
        this->merge_sort(record, 0, record.size() - 1);
        for (auto& r : record) {
            std::cout << r << " ";
        }
        std::cout << std::endl;
        return m_count;
    }

    void merge_sort(std::vector<int>& arr, int left, int right) {
        if (left >= right) {
            return;
        }
        int mid = left + (right - left) / 2;
        this->merge_sort(arr, left, mid);
        this->merge_sort(arr, mid + 1, right);
        this->merge(arr, left, mid, right);
    }

    void merge(std::vector<int>& nums, int left, int mid, int right) {
        // 辅助数组
        std::vector<int> tmp(right - left + 1);
        int left_idx = left;
        int right_idx = mid + 1;
        int tmp_idx = 0;

        // 比较元素
        while (left_idx <= mid && right_idx <= right) {
            // 将两个有序子序列中较小元素依次插入到结果数组中
            if (nums[left_idx] <= nums[right_idx]) {
                tmp[tmp_idx++] = nums[left_idx++];
            } else {
                // 存在逆序
                this->m_count += (mid - left_idx + 1);
                tmp[tmp_idx++] = nums[right_idx++];
            }
        }

        // 如果左侧有剩余元素，插入结果数组中
        while (left_idx <= mid) {
            tmp[tmp_idx++] = nums[left_idx++];
        }

        // 如果右侧有剩余元素，插入结果数组中
        while (right_idx <= right) {
             tmp[tmp_idx++] = nums[right_idx++];
        }

        // 复制tmp到nums
        for (int k = 0; k < tmp.size(); k++) {
            nums[left + k] = tmp[k];
        }
    }

private:
    int m_count = 0;
};