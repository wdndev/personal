/*
 * @lc app=leetcode.cn id=451 lang=cpp
 *
 * [451] 根据字符出现频率排序
 *
 * https://leetcode.cn/problems/sort-characters-by-frequency/description/
 *
 * algorithms
 * Medium (72.09%)
 * Likes:    506
 * Dislikes: 0
 * Total Accepted:    134K
 * Total Submissions: 185.8K
 * Testcase Example:  '"tree"'
 *
 * 给定一个字符串 s ，根据字符出现的 频率 对其进行 降序排序 。一个字符出现的 频率 是它出现在字符串中的次数。
 * 
 * 返回 已排序的字符串 。如果有多个答案，返回其中任何一个。
 * 
 * 
 * 
 * 示例 1:
 * 
 * 
 * 输入: s = "tree"
 * 输出: "eert"
 * 解释: 'e'出现两次，'r'和't'都只出现一次。
 * 因此'e'必须出现在'r'和't'之前。此外，"eetr"也是一个有效的答案。
 * 
 * 
 * 示例 2:
 * 
 * 
 * 输入: s = "cccaaa"
 * 输出: "cccaaa"
 * 解释: 'c'和'a'都出现三次。此外，"aaaccc"也是有效的答案。
 * 注意"cacaca"是不正确的，因为相同的字母必须放在一起。
 * 
 * 
 * 示例 3:
 * 
 * 
 * 输入: s = "Aabb"
 * 输出: "bbAa"
 * 解释: 此外，"bbaA"也是一个有效的答案，但"Aabb"是不正确的。
 * 注意'A'和'a'被认为是两种不同的字符。
 * 
 * 
 * 
 * 
 * 提示:
 * 
 * 
 * 1 <= s.length <= 5 * 10^5
 * s 由大小写英文字母和数字组成
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 哈希表 + 自定义排序 ：根据词频自定义排序
    string frequencySort1(string s) {
        // 哈希表统计字符出现次数
        std::unordered_map<char, int> map;
        for (const auto& ch : s) {
            map[ch]++;
        }

        // 定义列表，准备排序
        std::vector<std::pair<char, int>> vec;
        for (auto& it : map) {
            vec.push_back(it);
        }
        // 依据词频排序
        std::sort(vec.begin(), vec.end(), [](std::pair<char, int>& p1, std::pair<char, int>& p2){
            // 小顶堆是大于号
            return p1.second > p2.second;
        });

        std::string ans;
        for (auto& [ch, num] : vec) {
            for (int i = 0; i <num; i++) {
                ans.push_back(ch);
            }
        }

        return ans;
    }

    // 哈希表 + 优先队列
    string frequencySort(string s) {
        // 哈希表统计字符出现次数
        std::unordered_map<char, int> map;
        for (const auto& ch : s) {
            map[ch]++;
        }

        // 2.定义优先队列，将出现次数排序
        struct fre_comparison {
            bool operator() (std::pair<char, int>& p1, std::pair<char, int>& p2) {
                // 小顶堆是大于号
                return p1.second < p2.second;
            }
        };

        // 创建优先队列
        std::priority_queue<std::pair<char, int>, std::vector<std::pair<char, int>>, fre_comparison> pq;


        // 遍历 map 中的元素
        // 1.管他是啥，先入队列，队列会自己排序将他放在合适的位置
        for (const auto& elem : map) {
            pq.push(elem);
        }

        // 处理结果
        std::string ans;
        while (!pq.empty()) {
            auto [ch, num] = pq.top();
            for (int i = 0; i <num; i++) {
                ans.push_back(ch);
            }
            pq.pop();
        }

        return ans;
    }
};
// @lc code=end

