/*
 * @lc app=leetcode.cn id=347 lang=cpp
 *
 * [347] 前 K 个高频元素
 *
 * https://leetcode.cn/problems/top-k-frequent-elements/description/
 *
 * algorithms
 * Medium (63.53%)
 * Likes:    1745
 * Dislikes: 0
 * Total Accepted:    487.6K
 * Total Submissions: 767.5K
 * Testcase Example:  '[1,1,1,2,2,3]\n2'
 *
 * 给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。
 * 
 * 
 * 
 * 示例 1:
 * 
 * 
 * 输入: nums = [1,1,1,2,2,3], k = 2
 * 输出: [1,2]
 * 
 * 
 * 示例 2:
 * 
 * 
 * 输入: nums = [1], k = 1
 * 输出: [1]
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 
 * k 的取值范围是 [1, 数组中不相同的元素的个数]
 * 题目数据保证答案唯一，换句话说，数组中前 k 个高频元素的集合是唯一的
 * 
 * 
 * 
 * 
 * 进阶：你所设计算法的时间复杂度 必须 优于 O(n log n) ，其中 n 是数组大小。
 * 
 */

// @lc code=start
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
// @lc code=end

