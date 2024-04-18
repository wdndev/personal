/*
 * @lc app=leetcode.cn id=2007 lang=cpp
 *
 * [2007] 从双倍数组中还原原数组
 *
 * https://leetcode.cn/problems/find-original-array-from-doubled-array/description/
 *
 * algorithms
 * Medium (34.54%)
 * Likes:    59
 * Dislikes: 0
 * Total Accepted:    23K
 * Total Submissions: 55.6K
 * Testcase Example:  '[1,3,4,2,6,8]'
 *
 * 一个整数数组 original 可以转变成一个 双倍 数组 changed ，转变方式为将 original 中每个元素 值乘以 2
 * 加入数组中，然后将所有元素 随机打乱 。
 * 
 * 给你一个数组 changed ，如果 change 是 双倍 数组，那么请你返回 original数组，否则请返回空数组。original 的元素可以以
 * 任意 顺序返回。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 输入：changed = [1,3,4,2,6,8]
 * 输出：[1,3,4]
 * 解释：一个可能的 original 数组为 [1,3,4] :
 * - 将 1 乘以 2 ，得到 1 * 2 = 2 。
 * - 将 3 乘以 2 ，得到 3 * 2 = 6 。
 * - 将 4 乘以 2 ，得到 4 * 2 = 8 。
 * 其他可能的原数组方案为 [4,3,1] 或者 [3,1,4] 。
 * 
 * 
 * 示例 2：
 * 
 * 输入：changed = [6,3,0,1]
 * 输出：[]
 * 解释：changed 不是一个双倍数组。
 * 
 * 
 * 示例 3：
 * 
 * 输入：changed = [1]
 * 输出：[]
 * 解释：changed 不是一个双倍数组。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= changed.length <= 10^5
 * 0 <= changed[i] <= 10^5
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 排序 + 哈希表
    // 1.方便求最小值，把 changed 从小到大排序
    // 2.用哈希表cnt标记那些元素被删除
    // 3.遍历排序后的数组
    // 4.如果 x_i 不在cnt中，说明x不是某个元素的两倍，所以x在original中，将x加入答案，并标记 2x，即 cnt[2x]增加1.
    //   注意，无需标记x，从小到大遍历
    // 5.如果x在cnt中，说明x是某个元素的两倍，清除一个x的标记，即cnt[x]减一。如果减一后为0，从中删除x
    // 6.遍历结束，如果cnt不为空，则说明不是双倍，否则返回答案。
    vector<int> findOriginalArray(vector<int>& changed) {
        std::sort(changed.begin(), changed.end());
        std::unordered_multiset<int> mark;
        std::vector<int> ans;
        for (int x : changed) {
            // 在标记数组中查找
            auto it = mark.find(x);
            // 如果未找到
            if (it == mark.end()) {
                // 标记他的两倍
                mark.insert(2 * x);
                // 加入答案
                ans.push_back(x);
            } else {
                // 如果找到，清除一个标记
                mark.erase(it);
            }
        }

        // 只有所有双倍标记都被清除掉，才能说明 changed 是一个双倍数组
        return mark.empty() ? ans : std::vector<int>();
    }
};
// @lc code=end

