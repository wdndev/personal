/*
 * @lc app=leetcode.cn id=771 lang=cpp
 *
 * [771] 宝石与石头
 *
 * https://leetcode.cn/problems/jewels-and-stones/description/
 *
 * algorithms
 * Easy (85.64%)
 * Likes:    796
 * Dislikes: 0
 * Total Accepted:    212.1K
 * Total Submissions: 247.6K
 * Testcase Example:  '"aA"\n"aAAbbbb"'
 *
 *  给你一个字符串 jewels 代表石头中宝石的类型，另有一个字符串 stones 代表你拥有的石头。 stones
 * 中每个字符代表了一种你拥有的石头的类型，你想知道你拥有的石头中有多少是宝石。
 * 
 * 字母区分大小写，因此 "a" 和 "A" 是不同类型的石头。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：jewels = "aA", stones = "aAAbbbb"
 * 输出：3
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：jewels = "z", stones = "ZZ"
 * 输出：0
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= jewels.length, stones.length <= 50
 * jewels 和 stones 仅由英文字母组成
 * jewels 中的所有字符都是 唯一的
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 使用hash set判断
    int numJewelsInStones(string jewels, string stones) {
        int jewejs_num = 0;
        std::unordered_set<char> set;

        for (auto& j : jewels) {
            set.insert(j);
        }

        for (auto& s : stones) {
            if (set.count(s)) {
                jewejs_num++;
            }
        }

        return jewejs_num;
    }
};
// @lc code=end

