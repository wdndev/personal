// 给定一个字符串 s 和一个字符串数组 words。 words 中所有字符串 长度相同。
//  s 中的 串联子串 是指一个包含  words 中所有字符串以任意顺序排列连接起来的子串。
// 例如，如果 words = ["ab","cd","ef"]， 那么 "abcdef"， "abefcd"，"cdabef"， "cdefab"，"efabcd"， 和 "efcdab" 都是串联子串。 "acdbef" 不是串联子串，因为他不是任何 words 排列的连接。
// 返回所有串联子串在 s 中的开始索引。你可以以 任意顺序 返回答案。

// 示例 1：
// 输入：s = "barfoothefoobarman", words = ["foo","bar"]
// 输出：[0,9]
// 解释：因为 words.length == 2 同时 words[i].length == 3，连接的子字符串的长度必须为 6。
// 子串 "barfoo" 开始位置是 0。它是 words 中以 ["bar","foo"] 顺序排列的连接。
// 子串 "foobar" 开始位置是 9。它是 words 中以 ["foo","bar"] 顺序排列的连接。
// 输出顺序无关紧要。返回 [9,0] 也是可以的。

// 示例 2：
// 输入：s = "wordgoodgoodgoodbestword", words = ["word","good","best","word"]
// 输出：[]
// 解释：因为 words.length == 4 并且 words[i].length == 4，所以串联子串的长度必须为 16。
// s 中没有子串长度为 16 并且等于 words 的任何顺序排列的连接。
// 所以我们返回一个空数组。

// 示例 3：
// 输入：s = "barfoofoobarthefoobarman", words = ["bar","foo","the"]
// 输出：[6,9,12]
// 解释：因为 words.length == 3 并且 words[i].length == 3，所以串联子串的长度必须为 9。
// 子串 "foobarthe" 开始位置是 6。它是 words 中以 ["foo","bar","the"] 顺序排列的连接。
// 子串 "barthefoo" 开始位置是 9。它是 words 中以 ["bar","the","foo"] 顺序排列的连接。
// 子串 "thefoobar" 开始位置是 12。它是 words 中以 ["the","foo","bar"] 顺序排列的连接。
 
class Solution {
public:
    // words数组中：单词数量 word_num，每个单词的长度 word_len
    // s数组：长度：s_len
    // hash 表 diff表示窗口中单词的频次和words中单词频次之差
    // s数组的子单词出现，diff加1，出现words中的单词，相应的值减1
    // 窗口右移，同时对diff更新
    // 移动的过程中，如出现differ中值不为0的键数量为0，则表示窗口中的单词频次和words中单词频次一样
    // 窗口的左端点是一个代求的起始位置
    vector<int> findSubstring(string s, vector<string>& words) {
        std::vector<int> ans;
        // 单词个数
        int word_num = words.size();
        // 每个单词的长度
        int word_len = words[0].size();
        // s字符串长度
        int s_len = s.size();

        for (int i = 0; i < word_len && i + word_len * word_num <= s_len; i++) {
            std::unordered_map<std::string, int> differ;
            // 将s数组中的前word_num个单词hash表中
            for (int j = 0; j < word_num; j++) {
                differ[s.substr(i + j * word_len, word_len)]++;
            }
            // 在words数组中对比，是否一样
            for (auto& word : words) {
                if (--differ[word] == 0) {
                    differ.erase(word);
                }
            }

            for (int start = i; start < s_len - word_num * word_len + 1; start += word_len) {
                // 第一个数组已经在上面对比过了，应该跳过
                if (start != i) {
                    // 从s中获取子单词，加入hash表
                    std::string word = s.substr(start + (word_num - 1) * word_len, word_len);
                    if (++differ[word] == 0) {
                        differ.erase(word);
                    }

                    // 在words中，获取子单词，减去
                    word = s.substr(start - word_len, word_len);
                    if (--differ[word] == 0) {
                        differ.erase(word);
                    }
                }
                // 如果hash表为空，则代表对比成功，添加
                if (differ.empty()) {
                    ans.emplace_back(start);
                }
            }
        }

        return ans;
    }
};

