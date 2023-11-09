/*
 * @lc app=leetcode.cn id=49 lang=cpp
 *
 * [49] 字母异位词分组
 *
 * https://leetcode.cn/problems/group-anagrams/description/
 *
 * algorithms
 * Medium (67.76%)
 * Likes:    1686
 * Dislikes: 0
 * Total Accepted:    545K
 * Total Submissions: 804.3K
 * Testcase Example:  '["eat","tea","tan","ate","nat","bat"]'
 *
 * 给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。
 * 
 * 字母异位词 是由重新排列源单词的所有字母得到的一个新单词。
 * 
 * 
 * 
 * 示例 1:
 * 
 * 
 * 输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
 * 输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
 * 
 * 示例 2:
 * 
 * 
 * 输入: strs = [""]
 * 输出: [[""]]
 * 
 * 
 * 示例 3:
 * 
 * 
 * 输入: strs = ["a"]
 * 输出: [["a"]]
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= strs.length <= 10^4
 * 0 <= strs[i].length <= 100
 * strs[i] 仅包含小写字母
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.排序
    // 对两个字符串，排序之后得到的字符串一定是相同的，可以将排序后的字符串作为哈希表的键
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        std::unordered_map<std::string, std::vector<std::string>> map;

        for (auto& str : strs) {
            std::string key = str;
            std::sort(key.begin(), key.end());
            map[key].emplace_back(str);
        }

        std::vector<std::vector<std::string>> ans;
        for (auto it = map.begin(); it != map.end(); it++) {
            ans.emplace_back(it->second);
        }
        
        return ans;
    }

    // 2.hash 计数
    // 两个字符串中相同字母出现的次数一样，可以将每个字母出现的次数使用字符串表示，作为hash的键
    // 由于字符串只包含小写，使用长度为26的数组记录每个字母的出现次数。
    vector<vector<string>> groupAnagrams2(vector<string>& strs) {
        // 自定义 array<int, 26> 类型的哈希函数
        auto array_hash = [fn = hash<int>{}](const std::array<int, 26>& arr) -> size_t {
            return std::accumulate(arr.begin(), arr.end(), 0u, [&](size_t acc, int num) {
                return (acc << 1) ^ fn(num);
            });
        };

        std::unordered_map<std::array<int ,26>, std::vector<std::string>, decltype(array_hash)> map(0, array_hash);

        for (std::string& str : strs) {
            std::array<int ,26> counts{};
            int length = str.length();
            for (int i = 0; i < length; i++) {
                counts[str[i] - 'a']++;
            }
            map[counts].emplace_back(str);
        }

        std::vector<std::vector<std::string>> ans;
        for (auto it = map.begin(); it != map.end(); it++) {
            ans.emplace_back(it->second);
        }
        
        return ans;
    }
};
// @lc code=end

