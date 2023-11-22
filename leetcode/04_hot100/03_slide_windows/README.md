# 03滑动窗口

# 1.无重复字符的最长子串

[3. 无重复字符的最长子串 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/?envType=study-plan-v2\&envId=top-100-liked "3. 无重复字符的最长子串 - 力扣（LeetCode）")

```bash
给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。

示例 1:

输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
```

使用两个指针表示字符串中的某个子串（或窗口）的左右边界，其中左指针代表着上文中「枚举子串的起始位置」，而右指针即为上文中的 rk

在每一步的操作中，会将左指针向右移动一格，表示开始枚举下一个字符作为起始位置，然后可以不断地向右移动右指针，但需要保证这两个指针对应的子串中没有重复的字符。在移动结束后，这个子串就对应着 以左指针开始的，不包含重复字符的最长子串。我们记录下这个子串的长度；

在枚举结束后，找到的最长的子串的长度即为答案。

在上面的流程中，还需要使用一种数据结构来判断 **是否有重复的字符**，常用的数据结构为哈希集合（即 C++ 中的 std::unordered\_set，Java 中的 HashSet，Python 中的 set, JavaScript 中的 Set）。在左指针向右移动的时候，我们从哈希集合中移除一个字符，在右指针向右移动的时候，我们往哈希集合中添加一个字符。

```c++
class Solution {
public:
    // 滑动窗口 + hash
    int lengthOfLongestSubstring(string s) {
        // 哈希集合，记录每个字符是否出现过
        std::unordered_set<char> occ;
        int n = s.size();
        // 初始化右指针，初始值为-1，还没开始移动
        int right = -1;
        // 最长子串的长度
        int max_len = 0;

        // 开始遍历左指针位置
        for (int left = 0; left < n; left++) {
            if (left != 0) {
                // 左指针向右移动一格，一处一个字符
                occ.erase(s[left - 1]);
            }

            // 不断移动右指针，直到出现重复的字符，或是到达最后一个字符
            while (right + 1 < n && !occ.count(s[right + 1])) {
                occ.insert(s[right + 1]);
                right++;
            }

            // 更新max_len
            max_len = std::max(max_len, right - left + 1);
        }

        return max_len;
    }
};
```

# 2.找到字符串中所有字母的异位词

[438. 找到字符串中所有字母异位词 - 力扣（LeetCode）](https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/?envType=study-plan-v2\&envId=top-100-liked "438. 找到字符串中所有字母异位词 - 力扣（LeetCode）")

```bash
给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。

异位词 指由相同字母重排列形成的字符串（包括相同的字符串）。
```

需要在字符串 s 寻找字符串 p 的异位词。因为字符串 p 的异位词的长度一定与字符串 p 的长度相同，所以**可以在字符串 s 中构造一个长度为与字符串 p 的长度相同的滑动窗口，并在滑动中维护窗口中每种字母的数量；** 当窗口中每种字母的数量与字符串 p 中每种字母的数量相同时，则说明当前窗口为字符串 p 的异位词。

```c++
class Solution {
public:
    // 构造两个相同的滑动窗口，在滑动窗口中维护每个字母的数量
    // 如果字母数量相同，则说明是 异位词
    vector<int> findAnagrams(string s, string p) {
        int s_len = s.size();
        int p_len = p.size();

        // 如果s的长度小于p的，直接返回空
        if (s_len < p_len) {
            return {};
        }

        std::vector<int> ans;

        // s字符字母统计
        std::vector<int> s_count(26);
        // p字符字母统计
        std::vector<int> p_count(26);

        // 首先统计前p_len个字符串中的字符数量
        for (int i = 0; i < p_len; i++) {
            s_count[s[i] - 'a']++;
            p_count[p[i] - 'a']++;
        }

        // 如果最开始相同，加入0
        if (s_count == p_count) {
            ans.emplace_back(0);
        }

        // 再遍历s中剩余的字符串
        for (int i = 0; i < s_len - p_len; i++) {
            s_count[s[i] - 'a']--;
            s_count[s[i + p_len] - 'a']++;

            if (s_count == p_count) {
                ans.emplace_back(i + 1);
            }
        }

        return ans;
    }
};
```
