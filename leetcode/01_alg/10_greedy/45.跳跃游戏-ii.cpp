/*
 * @lc app=leetcode.cn id=45 lang=cpp
 *
 * [45] 跳跃游戏 II
 *
 * https://leetcode.cn/problems/jump-game-ii/description/
 *
 * algorithms
 * Medium (44.84%)
 * Likes:    2325
 * Dislikes: 0
 * Total Accepted:    571.7K
 * Total Submissions: 1.3M
 * Testcase Example:  '[2,3,1,1,4]'
 *
 * 给定一个长度为 n 的 0 索引整数数组 nums。初始位置为 nums[0]。
 * 
 * 每个元素 nums[i] 表示从索引 i 向前跳转的最大长度。换句话说，如果你在 nums[i] 处，你可以跳转到任意 nums[i + j]
 * 处:
 * 
 * 
 * 0 <= j <= nums[i] 
 * i + j < n
 * 
 * 
 * 返回到达 nums[n - 1] 的最小跳跃次数。生成的测试用例可以到达 nums[n - 1]。
 * 
 * 
 * 
 * 示例 1:
 * 
 * 
 * 输入: nums = [2,3,1,1,4]
 * 输出: 2
 * 解释: 跳到最后一个位置的最小跳跃数是 2。
 * 从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
 * 
 * 
 * 示例 2:
 * 
 * 
 * 输入: nums = [2,3,0,1,4]
 * 输出: 2
 * 
 * 
 * 
 * 
 * 提示:
 * 
 * 
 * 1 <= nums.length <= 10^4
 * 0 <= nums[i] <= 1000
 * 题目保证可以到达 nums[n-1]
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.如果某一个作为 起跳点 的格子可以跳跃的距离是 3，
    //   那么表示后面 3 个格子都可以作为 起跳点。
    //   可以对每一个能作为 起跳点 的格子都尝试跳一次，把 能跳到最远的距离 不断更新。

    // 2.如果从这个 起跳点 起跳叫做第 1 次 跳跃，
    //   那么从后面 3 个格子起跳 都 可以叫做第 2 次 跳跃。

    // 3.所以，当一次 跳跃 结束时，从下一个格子开始，到现在 能跳到最远的距离，都 是下一次 跳跃 的 起跳点。 
    //   对每一次 跳跃 用 for 循环来模拟。
    //   跳完一次之后，更新下一次 起跳点 的范围。
    //   在新的范围内跳，更新 能跳到最远的距离。

    // 4.记录 跳跃 次数，如果跳到了终点，就得到了结果。

    int jump(vector<int>& nums) {
        int ans = 0;
        int start = 0;
        int end = 1;

        while (end < nums.size()) {
            int max_pos = 0;
            for (int i = start; i < end; i++) {
                // 能跳到的最远距离
                max_pos = std::max(max_pos, i + nums[i]);
            }

            // 下次起跳点范围开始的格子
            start = end;
            // 下次起跳点范围结束的格子
            end = max_pos + 1;
            // 跳跃次数
            ans++;
        }

        return ans;
    }
};
// @lc code=end

