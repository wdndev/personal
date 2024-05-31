/*
 * @lc app=leetcode.cn id=128 lang=cpp
 *
 * [128] 最长连续序列
 *
 * https://leetcode.cn/problems/longest-consecutive-sequence/description/
 *
 * algorithms
 * Medium (51.80%)
 * Likes:    2077
 * Dislikes: 0
 * Total Accepted:    632.9K
 * Total Submissions: 1.2M
 * Testcase Example:  '[100,4,200,1,3,2]'
 *
 * 给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
 * 
 * 请你设计并实现时间复杂度为 O(n) 的算法解决此问题。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [100,4,200,1,3,2]
 * 输出：4
 * 解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [0,3,7,2,5,8,4,6,0,1]
 * 输出：9
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 0 
 * -10^9 
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 哈希表
    // 每个数都判断一次这个数是不是连续序列的开头那个数
    // - 使用哈希表查找这个数前面一个数是否存在，即 num-1 这个数在序列中是否存在。如果存在，肯定不是开头，跳过
    // - 因此只要对每个开头的数进行循环，直到这个数不再连续
    // 举例：[100, 4, 200, 1, 3, 4, 2] -> 去重 -> [100, 4, 200, 1, 3, 2]
    // - 元素100是开头,因为没有99，且以100开头的序列长度为1
    // - 元素4不是开头，因为有3存在，过，
    // - 元素200是开头，因为没有199，且以200开头的序列长度为1
    // - 元素1是开头，因为没有0，且以1开头的序列长度为4，因为依次累加，2，3，4都存在。
    // - 元素3不是开头，因为2存在，过，
    // - 元素2不是开头，因为1存在，过。
    int longestConsecutive(vector<int>& nums) {
        std::unordered_set<int> num_set;
        // 哈希去重
        for (auto& n : nums) {
            num_set.insert(n);
        }

        int max_len = 0;

        for (auto& n : num_set) {
            // 判断 n-1 是否存在，如果存在，继续判断
            // 不存在，直接跳过
            if (!num_set.count(n - 1)) {
                int curr_n = n;
                int curr_len = 1;

                while (num_set.count(curr_n + 1)) {
                    curr_n++;
                    curr_len++;
                }

                max_len = std::max(max_len, curr_len);
            }
        }

        return max_len;
    }
};
// @lc code=end

