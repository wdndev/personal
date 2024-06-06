/*
 * @lc app=leetcode.cn id=2938 lang=cpp
 *
 * [2938] 区分黑球与白球
 *
 * https://leetcode.cn/problems/separate-black-and-white-balls/description/
 *
 * algorithms
 * Medium (54.27%)
 * Likes:    40
 * Dislikes: 0
 * Total Accepted:    23.2K
 * Total Submissions: 36K
 * Testcase Example:  '"101"'
 *
 * 桌子上有 n 个球，每个球的颜色不是黑色，就是白色。
 * 
 * 给你一个长度为 n 、下标从 0 开始的二进制字符串 s，其中 1 和 0 分别代表黑色和白色的球。
 * 
 * 在每一步中，你可以选择两个相邻的球并交换它们。
 * 
 * 返回「将所有黑色球都移到右侧，所有白色球都移到左侧所需的 最小步数」。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：s = "101"
 * 输出：1
 * 解释：我们可以按以下方式将所有黑色球移到右侧：
 * - 交换 s[0] 和 s[1]，s = "011"。
 * 最开始，1 没有都在右侧，需要至少 1 步将其移到右侧。
 * 
 * 示例 2：
 * 
 * 
 * 输入：s = "100"
 * 输出：2
 * 解释：我们可以按以下方式将所有黑色球移到右侧：
 * - 交换 s[0] 和 s[1]，s = "010"。
 * - 交换 s[1] 和 s[2]，s = "001"。
 * 可以证明所需的最小步数为 2 。
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：s = "0111"
 * 输出：0
 * 解释：所有黑色球都已经在右侧。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= n == s.length <= 10^5
 * s[i] 不是 '0'，就是 '1'。
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 累加每个 0 左边的 1 的个数
    // 将所有黑色球都移到右侧，所有白色球都移到左侧；也就是 0 都在 1 的左边
    // 遍历 s 的同时，统计 1 的个数 cnt_1。
    // 遇到 0 时，这个 0 和它左边 cnt_1个 1 都要交换，把 cnt_1 加入答案
    long long minimumSteps(string s) {
        long long ans = 0;
        int cnt_1 = 0;

        for (char c : s) {
            if (c == '1') {
                cnt_1++;
            } else {
                ans += cnt_1;
            }
        }

        return ans;
    }
};
// @lc code=end

