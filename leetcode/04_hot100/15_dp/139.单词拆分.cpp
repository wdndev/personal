/*
 * @lc app=leetcode.cn id=139 lang=cpp
 *
 * [139] 单词拆分
 *
 * https://leetcode.cn/problems/word-break/description/
 *
 * algorithms
 * Medium (54.64%)
 * Likes:    2352
 * Dislikes: 0
 * Total Accepted:    508K
 * Total Submissions: 929.7K
 * Testcase Example:  '"leetcode"\n["leet","code"]'
 *
 * 给你一个字符串 s 和一个字符串列表 wordDict 作为字典。请你判断是否可以利用字典中出现的单词拼接出 s 。
 * 
 * 注意：不要求字典中出现的单词全部都使用，并且字典中的单词可以重复使用。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入: s = "leetcode", wordDict = ["leet", "code"]
 * 输出: true
 * 解释: 返回 true 因为 "leetcode" 可以由 "leet" 和 "code" 拼接成。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入: s = "applepenapple", wordDict = ["apple", "pen"]
 * 输出: true
 * 解释: 返回 true 因为 "applepenapple" 可以由 "apple" "pen" "apple" 拼接成。
 * 注意，你可以重复使用字典中的单词。
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
 * 输出: false
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= s.length <= 300
 * 1 <= wordDict.length <= 1000
 * 1 <= wordDict[i].length <= 20
 * s 和 wordDict[i] 仅由小写英文字母组成
 * wordDict 中的所有字符串 互不相同
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.动态规划
    bool wordBreak(string s, vector<string>& wordDict) {
        int str_len = s.size();
        // 通过看能否通过修改让最后一个元素的值为true，如果可以，则返回true，否则返回false;
        std::vector<bool> dp(str_len + 1, false);
        dp[0] = true;

        // 遍历字符串s
        for (int i = 0; i < str_len; i++) {
            // 如果前i位不能表示为word中的某一个单词，则跳过
            if (!dp[i]) {
                continue;
            }
            // 遍历单词
            // 此处有for循环，可能进行多处修改，修改为true地方我们后续当i遍历到该位置时要继续进行判定，
            // 如果有一次连续修改使得dp[len]成功改成了true,则直接break并返回true；
            // 也就是说，i遍历到每个dp[i]==true的位置，都有机会将dp[len]修改为true，
            // 如果遍历完全都没能把dp[len]修改为true，则说明无法成功拼接出字符串s
            for (auto& word : wordDict) {
                if (word.size() + i <= str_len && s.substr(i, word.size()) == word) {
                    dp[i + word.size()] = true;
                }
            }
        }
        return dp[str_len];
    }

    // 2.记忆化搜索 + dfs
    bool wordBreak2(string s, vector<string>& wordDict) {
        std::unordered_set<std::string> uset;
        for (auto& w : wordDict) {
            uset.insert(w);
        }
        // 使用一个数组来记录从每个索引位置开始的子问题是否可解
        std::vector<int> memo(s.size(), 0);

        return this->dfs(s, uset, 0, memo);
    }

    bool dfs(std::string s, std::unordered_set<std::string>& uset, int idx, std::vector<int>& memo) {
        if (idx >= s.size()) {
            return true;
        }

        if (memo[idx] != 0) {
            return memo[idx] == 1;
        }

        std::string tmp_str = "";
        for (int i = idx; i < s.size(); i++) {
            tmp_str = tmp_str + s[i];
            // 如果字典包含当前的词语，且切割下去可行，标记为true，并返回
            if (uset.find(tmp_str) != uset.end() && this->dfs(s, uset, i + 1, memo)) {
                memo[idx] = 1;
                return true;
            }
        }

        // 标记从当前位置切割不行
        memo[idx] = -1;

        return false;
    }
};
// @lc code=end

