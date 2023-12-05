# 13 heap

# 1.数组中的第k个最大元素

[215. 数组中的第K个最大元素 - 力扣（LeetCode）](https://leetcode.cn/problems/kth-largest-element-in-an-array/description/?envType=study-plan-v2\&envId=top-100-liked "215. 数组中的第K个最大元素 - 力扣（LeetCode）")

```bash
给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。

请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

你必须设计并实现时间复杂度为 O(n) 的算法解决此问题。


示例 1:

输入: [3,2,1,5,6,4], k = 2
输出: 5
示例 2:

输入: [3,2,3,1,2,4,5,5,6], k = 4
输出: 4
```

1.  快排：先对原数组排序，再返回倒数第 k 个位置，这样平均时间复杂度是 $O(n \log n)$，但其实可以做的更快。

在分解的过程当中，我们会对子数组进行划分，如果划分得到的 q 正好就是我们需要的下标，就直接返回 $a[q]$；否则，如果 q 比目标下标小，就递归右子区间，否则递归左子区间。这样就可以把原来递归两个区间变成只递归一个区间，提高了时间效率。这就是「快速选择」算法。

```c++
class Solution {
public:
    // 1.快速排序
    int findKthLargest(vector<int>& nums, int k) {
        return this->quick_select(nums, 0, nums.size() - 1, nums.size() - k);
    }
    int quick_select(std::vector<int>& nums, int left, int right, int k) {
        if (left == right) {
            return nums[k];
        }

        int pivot = nums[left];
        int i = left - 1;
        int j = right + 1;
        while (i < j)
        {
            do {
                i++;
            } while (nums[i] < pivot);

            do {
                j--;
            } while (nums[j] > pivot);

            if (i < j) {
                swap(nums[i], nums[j]);
            }
        }
        if (k <= j) {
            return this->quick_select(nums, left, j, k);
        } else {
            return this->quick_select(nums, j + 1, right, k);
        }
    }
};
```

1.  建立一个大根堆，做`k - 1`次删除操作后堆顶元素就是我们要找的答案。

```c++
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        int heap_size = nums.size();
        this->build_max_heap(nums, heap_size);
        for (int i = nums.size() - 1; i >= nums.size() - k + 1; i--) {
            swap(nums[0], nums[i]);
            heap_size--;
            this->max_heapify(nums, 0, heap_size);
        }

        return nums[0];
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
```

# 2.前k个高频元素

[347. 前 K 个高频元素 - 力扣（LeetCode）](https://leetcode.cn/problems/top-k-frequent-elements/description/?envType=study-plan-v2\&envId=top-100-liked "347. 前 K 个高频元素 - 力扣（LeetCode）")

```bash
给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。


示例 1:

输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]
示例 2:

输入: nums = [1], k = 1
输出: [1]
```

```c++
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        // 1.map记录元素出现的次数 <元素，出现次数>
        std::unordered_map<int, int> map;
        for (const auto& n : nums) {
            map[n]++;
        }

        // 2.定义优先队列，将出现次数排序
        // 自定义优先队列方式，小顶堆
        struct fre_comparison {
            bool operator() (std::pair<int, int>& p1, std::pair<int, int>& p2) {
                // 小顶堆是大于号
                return p1.second > p2.second;
            }
        };

        // 创建优先队列
        std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, fre_comparison> pq;
        // 遍历 map 中的元素
        // 1.管他是啥，先入队列，队列会自己排序将他放在合适的位置
        // 2.若队列元素个数超过k，则间栈顶元素出栈
        for (const auto& elem : map) {
            pq.push(elem);
            if (pq.size() > k) {
                pq.pop();
            }
        }

        // 将结果到处
        std::vector<int> ans;
        while (!pq.empty()) {
            ans.push_back(pq.top().first);
            pq.pop();
        }

        return ans;
    }
};
```

## 优先队列

#### 队列定义

`priority_queue<Type, Container, Functional>;`

-   `Type`是要存放的数据类型
-   `Container`是实现底层堆的容器，必须是数组实现的容器，如vector、deque
-   `Functional`是比较方式/比较函数/优先级

`priority_queue<Type>;`此时默认的容器是vector，默认的比较方式是大顶堆`less<type>`

示例

```c++
//小顶堆
priority_queue <int,vector<int>,greater<int> > q;
//大顶堆
priority_queue <int,vector<int>,less<int> >q;
//默认大顶堆
priority_queue<int> a;

```

```c++
//pair
priority_queue<pair<int, int> > a;
pair<int, int> b(1, 2);
pair<int, int> c(1, 3);
pair<int, int> d(2, 5);
a.push(d);
a.push(c);
a.push(b);
while (!a.empty()) 
{
   cout << a.top().first << ' ' << a.top().second << '\n';
   a.pop();
}
//输出结果为：
2 5
1 3
1 2

```

#### 自定义比较方式

当数据类型并不是基本数据类型，而是自定义的数据类型时，就不能用greater或less的比较方式了，而是需要自定义比较方式

在此假设数据类型是自定义的水果：

```c++
struct fruit {
  string name;
  int price;
};
```

有两种自定义比较方式的方法，如下

##### (1) 重载运算符

```c++
// 若希望水果价格高为优先级高，则
//大顶堆
struct fruit {
  string name;
  int price;
  friend bool operator < (fruit f1,fruit f2) {
    return f1.peice < f2.price;
  }
};


// 若希望水果价格低为优先级高
//小顶堆
struct fruit {
  string name;
  int price;
  friend bool operator < (fruit f1,fruit f2) {
    return f1.peice > f2.price;  //此处是>
  }
};

```

##### (2) 仿函数

若希望水果价格高为优先级高，则

```c++
//大顶堆
struct myComparison {
  bool operator () (fruit f1,fruit f2) {
    return f1.price < f2.price;
  }
};

//此时优先队列的定义应该如下
priority_queue<fruit,vector<fruit>,myComparison> q;
```

# 3.数据流的中位数

[295. 数据流的中位数 - 力扣（LeetCode）](https://leetcode.cn/problems/find-median-from-data-stream/description/?envType=study-plan-v2\&envId=top-100-liked "295. 数据流的中位数 - 力扣（LeetCode）")

```c++
中位数是有序整数列表中的中间值。如果列表的大小是偶数，则没有中间值，中位数是两个中间值的平均值。

- 例如 arr = [2,3,4] 的中位数是 3 。
- 例如 arr = [2,3] 的中位数是 (2 + 3) / 2 = 2.5 。

实现 MedianFinder 类:

- MedianFinder() 初始化 MedianFinder 对象。
- void addNum(int num) 将数据流中的整数 num 添加到数据结构中。
- double findMedian() 返回到目前为止所有元素的中位数。与实际答案相差 10-5 以内的答案将被接受。

示例 1：

输入
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
输出
[null, null, null, 1.5, null, 2.0]

解释
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // 返回 1.5 ((1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0
```

用两个优先队列 `que_max `和 `que_min `分别记录大于中位数的数和小于等于中位数的数。当累计添加的数的数量为奇数时，`que_min `中的数的数量比 `que_max  `多一个，此时中位数为 que\_min  的队头。当累计添加的数的数量为偶数时，两个优先队列中的数的数量相同，此时中位数为它们的队头的平均值。

当尝试添加一个数 `num `到数据结构中，需要分情况讨论：

1.  `num≤max⁡{que_min}` ：此 时 num 小于等于中位数，我们需要将该数添加到 `que_min `中。新的中位数将小于等于原来的中位数，因此我们可能需要将 `que_min`中最大的数移动到 `que_max `中。
2.  `num>max⁡{que_min}` ：此时 num 大于中位数，我们需要将该数添加到 `que_min `中。新的中位数将大于等于原来的中位数，因此我们可能需要将 `que_max `中最小的数移动到 `que_min ` 中。

特别地，当累计添加的数的数量为 0 时，将 `num`添加到 `que_min`中。

```c++
class MedianFinder {
public:
    MedianFinder() {

    }
    
    void addNum(int num) {
        if (m_que_min.empty() || num <= m_que_min.top()) {
            m_que_min.push(num);
            if (m_que_max.size() + 1 < m_que_min.size()) {
                m_que_max.push(m_que_min.top());
                m_que_min.pop();
            }
        } else {
            m_que_max.push(num);
            if (m_que_max.size() > m_que_min.size()) {
                m_que_min.push(m_que_max.top());
                m_que_max.pop();
            }
        }
    }
    
    double findMedian() {
        if (m_que_min.size() > m_que_max.size()) {
            return m_que_min.top();
        }

        return (m_que_min.top() + m_que_max.top()) / 2.0;
    }
private:
    std::priority_queue<int, std::vector<int>, less<int>> m_que_min;
    std::priority_queue<int, std::vector<int>, greater<int>> m_que_max;
};
```
