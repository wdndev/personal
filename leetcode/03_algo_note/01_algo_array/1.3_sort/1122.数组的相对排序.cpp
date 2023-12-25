/*
 * @lc app=leetcode.cn id=1122 lang=cpp
 *
 * [1122] 数组的相对排序
 *
 * https://leetcode.cn/problems/relative-sort-array/description/
 *
 * algorithms
 * Easy (70.55%)
 * Likes:    287
 * Dislikes: 0
 * Total Accepted:    91.6K
 * Total Submissions: 129.7K
 * Testcase Example:  '[2,3,1,3,2,4,6,7,9,2,19]\n[2,1,4,3,9,6]'
 *
 * 给你两个数组，arr1 和 arr2，arr2 中的元素各不相同，arr2 中的每个元素都出现在 arr1 中。
 * 
 * 对 arr1 中的元素进行排序，使 arr1 中项的相对顺序和 arr2 中的相对顺序相同。未在 arr2 中出现过的元素需要按照升序放在 arr1
 * 的末尾。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：arr1 = [2,3,1,3,2,4,6,7,9,2,19], arr2 = [2,1,4,3,9,6]
 * 输出：[2,2,2,1,4,3,3,9,6,7,19]
 * 
 * 
 * 示例  2:
 * 
 * 
 * 输入：arr1 = [28,6,22,8,44,17], arr2 = [22,28,8,6]
 * 输出：[22,28,8,6,17,44]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= arr1.length, arr2.length <= 1000
 * 0 <= arr1[i], arr2[i] <= 1000
 * arr2 中的元素 arr2[i]  各不相同 
 * arr2 中的每个元素 arr2[i] 都出现在 arr1 中
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    vector<int> relativeSortArray1(vector<int>& arr1, vector<int>& arr2) {
        std::unordered_map<int, int> rank;
        for (int i = 0; i < arr2.size(); i++) {
            rank[arr2[i]] = i;
        }

        std::sort(arr1.begin(), arr1.end(), [&](int x, int y) {
            // 两个都出现在哈希表中，比较他们对应的值
            if (rank.count(x)) {
                return rank.count(y) ? rank[x] < rank[y] : true;
            // 不在哈希表中，比较本省
            } else {
                return rank.count(y) ? false : x < y;
            }
        });

        return arr1;
    }

    vector<int> relativeSortArray(vector<int>& arr1, vector<int>& arr2) {
        // 计算待排序序列中最大元素
        int upper = *max_element(arr1.begin(), arr1.end());
        // 统计数组
        std::vector<int> counts(upper + 1);
        for (int x: arr1) {
            counts[x]++;
        }
        std::vector<int> ans;
        for (int x: arr2) {
            for (int i = 0; i < counts[x]; i++) {
                ans.push_back(x);
            }
            counts[x] = 0;
        }
        for (int x = 0; x <= upper; x++) {
            for (int i = 0; i < counts[x]; i++) {
                ans.push_back(x);
            }
        }
        return ans;
    }

};
// @lc code=end

