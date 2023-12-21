/*
 * @lc app=leetcode.cn id=881 lang=cpp
 *
 * [881] 救生艇
 *
 * https://leetcode.cn/problems/boats-to-save-people/description/
 *
 * algorithms
 * Medium (54.00%)
 * Likes:    290
 * Dislikes: 0
 * Total Accepted:    71.8K
 * Total Submissions: 132.8K
 * Testcase Example:  '[1,2]\n3'
 *
 * 给定数组 people 。people[i]表示第 i^ 个人的体重 ，船的数量不限，每艘船可以承载的最大重量为 limit。
 * 
 * 每艘船最多可同时载两人，但条件是这些人的重量之和最多为 limit。
 * 
 * 返回 承载所有人所需的最小船数 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：people = [1,2], limit = 3
 * 输出：1
 * 解释：1 艘船载 (1, 2)
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：people = [3,2,2,1], limit = 3
 * 输出：3
 * 解释：3 艘船分别载 (1, 2), (2) 和 (3)
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：people = [3,5,3,4], limit = 5
 * 输出：4
 * 解释：4 艘船分别载 (3), (3), (4), (5)
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= people.length <= 5 * 10^4
 * 1 <= people[i] <= limit <= 3 * 10^4
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 贪心算法 + 双指针
    // 排序后，让最最重的和最轻的先走
    int numRescueBoats(vector<int>& people, int limit) {
        // 排序
        std::sort(people.begin(), people.end());
        int left = 0;
        int right = people.size() - 1;
        int ans = 0;

        while (left < right) {
            if (people[left] + people[right] > limit) {
                right--;
            } else {
                left++;
                right--;
            }
            ans++;
        }
        // 处理最后一个
        if (left == right) {
            ans++;
        }

        return ans;
    }
};
// @lc code=end

