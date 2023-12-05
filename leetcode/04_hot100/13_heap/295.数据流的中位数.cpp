/*
 * @lc app=leetcode.cn id=295 lang=cpp
 *
 * [295] 数据流的中位数
 *
 * https://leetcode.cn/problems/find-median-from-data-stream/description/
 *
 * algorithms
 * Hard (53.76%)
 * Likes:    911
 * Dislikes: 0
 * Total Accepted:    122.1K
 * Total Submissions: 227.2K
 * Testcase Example:  '["MedianFinder","addNum","addNum","findMedian","addNum","findMedian"]\n' +
  '[[],[1],[2],[],[3],[]]'
 *
 * 中位数是有序整数列表中的中间值。如果列表的大小是偶数，则没有中间值，中位数是两个中间值的平均值。
 * 
 * 
 * 例如 arr = [2,3,4] 的中位数是 3 。
 * 例如 arr = [2,3] 的中位数是 (2 + 3) / 2 = 2.5 。
 * 
 * 
 * 实现 MedianFinder 类:
 * 
 * 
 * 
 * MedianFinder() 初始化 MedianFinder 对象。
 * 
 * 
 * void addNum(int num) 将数据流中的整数 num 添加到数据结构中。
 * 
 * 
 * double findMedian() 返回到目前为止所有元素的中位数。与实际答案相差 10^-5 以内的答案将被接受。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入
 * ["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
 * [[], [1], [2], [], [3], []]
 * 输出
 * [null, null, null, 1.5, null, 2.0]
 * 
 * 解释
 * MedianFinder medianFinder = new MedianFinder();
 * medianFinder.addNum(1);    // arr = [1]
 * medianFinder.addNum(2);    // arr = [1, 2]
 * medianFinder.findMedian(); // 返回 1.5 ((1 + 2) / 2)
 * medianFinder.addNum(3);    // arr[1, 2, 3]
 * medianFinder.findMedian(); // return 2.0
 * 
 * 提示:
 * 
 * 
 * -10^5 <= num <= 10^5
 * 在调用 findMedian 之前，数据结构中至少有一个元素
 * 最多 5 * 10^4 次调用 addNum 和 findMedian
 * 
 * 
 */

// @lc code=start
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

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder* obj = new MedianFinder();
 * obj->addNum(num);
 * double param_2 = obj->findMedian();
 */
// @lc code=end

