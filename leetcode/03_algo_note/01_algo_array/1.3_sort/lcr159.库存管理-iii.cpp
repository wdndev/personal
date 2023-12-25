
// [LCR 159. 库存管理 III - 力扣（LeetCode）](https://leetcode.cn/problems/zui-xiao-de-kge-shu-lcof/description/)

// ```C++
// 仓库管理员以数组 stock 形式记录商品库存表，其中 stock[i] 表示对应商品库存余量。请返回库存余量最少的 cnt 个商品余量，返回 顺序不限。

 

// 示例 1：

// 输入：stock = [2,5,7,4], cnt = 1
// 输出：[2]
// 示例 2：

// 输入：stock = [0,2,3,6], cnt = 2
// 输出：[0,2] 或 [2,0]
```

class Solution {
public:
    // 1.对原数组排序，取出前n个
    vector<int> inventoryManagement(vector<int>& stock, int cnt) {
        std::vector<int> ans(cnt, 0);
        std::sort(stock.begin(), stock.end());
        for (int i = 0; i < cnt; i++) {
            ans[i] = stock[i];
        }
        return ans;
    }
};

class Solution {
public:
    // 2.堆排序，取出前n个
    vector<int> inventoryManagement(vector<int>& stock, int cnt) {
        int heap_size = stock.size();
        this->build_max_heap(stock, heap_size);
        for (int i = stock.size() - 1; i >= 0; i--) {
            swap(stock[0], stock[i]);
            heap_size--;
            this->max_heapify(stock, 0, heap_size);
        }

        std::vector<int> ans(cnt, 0);
        for (int i = 0; i < cnt; i++) {
            ans[i] = stock[i];
        }
        return ans;
    }
    
    // 从上往下堆化
    void max_heapify(std::vector<int>& nums, int i, int heap_size) {
        // 父节点，左子树，右子树
        int largest_idx = i;
        int left_idx = 2 * i + 1;
        int right_idx = 2 * i + 2;
        
        // 寻找左右子树两个子节点中最大的一个
        if (left_idx < heap_size && nums[left_idx] > nums[largest_idx]) {
            largest_idx = left_idx;
        }
        if (right_idx < heap_size && nums[right_idx] > nums[largest_idx]) {
            largest_idx = right_idx;
        }
        // 如果能找到子节点比自己大
        // 交换元素，继续堆化
        if (largest_idx != i) {
            swap(nums[i], nums[largest_idx]);
            this->max_heapify(nums, largest_idx, heap_size);
        }
    }

    void build_max_heap(std::vector<int>& nums, int heap_size) {
        for (int i = heap_size / 2; i >= 0; i--) {
            this->max_heapify(nums, i, heap_size);
        }
    }
};


