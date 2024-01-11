
// 给定一个窗口大小和一个整数数据流，根据该滑动窗口的大小，计算滑动窗口里所有数字的平均值。

// 实现 MovingAverage 类：

// - MovingAverage(int size) 用窗口大小 size 初始化对象。
// - double next(int val) 成员函数 next 每次调用的时候都会往滑动窗口增加一个整数，请计算并返回数据流中最后 size 个值的移动平均值，即滑动窗口里所有数字的平均值。
 

// 示例：

// 输入：
// inputs = ["MovingAverage", "next", "next", "next", "next"]
// inputs = [[3], [1], [10], [3], [5]]
// 输出：
// [null, 1.0, 5.5, 4.66667, 6.0]

// 解释：
// MovingAverage movingAverage = new MovingAverage(3);
// movingAverage.next(1); // 返回 1.0 = 1 / 1
// movingAverage.next(10); // 返回 5.5 = (1 + 10) / 2
// movingAverage.next(3); // 返回 4.66667 = (1 + 10 + 3) / 3
// movingAverage.next(5); // 返回 6.0 = (10 + 3 + 5) / 3

// 1. 使用队列保存滑动窗口的元素，并记录对应窗口大小和元素和。
// 2. 当队列长度小于窗口大小的时候，直接向队列中添加元素，并记录当前窗口中的元素和。
// 3. 当队列长度等于窗口大小的时候，先将队列头部元素弹出，再添加元素，并记录当前窗口中的元素和。
// 4. 然后根据元素和和队列中元素个数计算出平均值。


class MovingAverage {
public:
    /** Initialize your data structure here. */
    MovingAverage(int size) {
        m_size = size;
        m_sum = 0.0;
    }
    
    double next(int val) {
        if (m_queue.size() < m_size) {
            m_queue.push(val);
        } else {
            if (!m_queue.empty()) {
                m_sum -= m_queue.front();
                m_queue.pop();
            }
            m_queue.push(val);
        }
        m_sum += val;

        return m_sum / m_queue.size();
    }
private:
    std::queue<int> m_queue;
    int m_size;
    double m_sum;
};


