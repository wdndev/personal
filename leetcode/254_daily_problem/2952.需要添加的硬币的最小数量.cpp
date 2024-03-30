/*
 * @lc app=leetcode.cn id=2952 lang=cpp
 *
 * [2952] 需要添加的硬币的最小数量
 *
 * https://leetcode.cn/problems/minimum-number-of-coins-to-be-added/description/
 *
 * algorithms
 * Medium (50.53%)
 * Likes:    85
 * Dislikes: 0
 * Total Accepted:    14.7K
 * Total Submissions: 22.8K
 * Testcase Example:  '[1,4,10]\n19'
 *
 * 给你一个下标从 0 开始的整数数组 coins，表示可用的硬币的面值，以及一个整数 target 。
 * 
 * 如果存在某个 coins 的子序列总和为 x，那么整数 x 就是一个 可取得的金额 。
 * 
 * 返回需要添加到数组中的 任意面值 硬币的 最小数量 ，使范围 [1, target] 内的每个整数都属于 可取得的金额 。
 * 
 * 数组的 子序列 是通过删除原始数组的一些（可能不删除）元素而形成的新的 非空 数组，删除过程不会改变剩余元素的相对位置。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：coins = [1,4,10], target = 19
 * 输出：2
 * 解释：需要添加面值为 2 和 8 的硬币各一枚，得到硬币数组 [1,2,4,8,10] 。
 * 可以证明从 1 到 19 的所有整数都可由数组中的硬币组合得到，且需要添加到数组中的硬币数目最小为 2 。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：coins = [1,4,10,5,7,19], target = 19
 * 输出：1
 * 解释：只需要添加一枚面值为 2 的硬币，得到硬币数组 [1,2,4,5,7,10,19] 。
 * 可以证明从 1 到 19 的所有整数都可由数组中的硬币组合得到，且需要添加到数组中的硬币数目最小为 1 。
 * 
 * 示例 3：
 * 
 * 
 * 输入：coins = [1,1,1], target = 20
 * 输出：3
 * 解释：
 * 需要添加面值为 4 、8 和 16 的硬币各一枚，得到硬币数组 [1,1,1,4,8,16] 。 
 * 可以证明从 1 到 20 的所有整数都可由数组中的硬币组合得到，且需要添加到数组中的硬币数目最小为 3 。
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= target <= 10^5
 * 1 <= coins.length <= 10^5
 * 1 <= coins[i] <= target
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 归纳法
    // 假设现在得到了区间[0, s-1]中的所有整数，如果此时遍历到整数 x = coins[i]，
    // 那么把 [0, s-1]中的每个整数增加x，就得到 [x, s + x -1]中所有的数组
    // 
    // 把 coins 从小到大排序，遍历 x = coins[i]，分类讨论，看是否需要添加数字
    // 1. 如果 x <= s, 那么合并 [0, s-1]和[x, s+x-1]这两个区间，可以得到[0, s+x-1]中所有整数
    // 2. 如果 x > s，或者遍历完 coins数组，这意味着我们无法得到s，那么就要把s加入到数组中，
    //    这样就得到 [s, 2s - 1]中所有的整数，再与[0, s-1]合并，可以得到[0, 2s - 1]。
    //    然后再考虑x和2s的大小关系
    // 3.当s > target时，就得到所有整数，退出训练
    int minimumAddedCoins(vector<int>& coins, int target) {
        std::sort(coins.begin(), coins.end());
        int ans = 0;
        int s = 1;
        int i = 0;
        while (s <= target) {
            if (i < coins.size() && coins[i] <= s) {
                s += coins[i];
                i++;
            } else {
                // 必须添加s
                s *= 2;
                ans++;
            }
        }

        return ans;
    }
};
// @lc code=end

