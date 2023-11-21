# 01 哈希

# 1.两数之和

[1. 两数之和 - 力扣（LeetCode）](https://leetcode.cn/problems/two-sum/description/?envType=study-plan-v2\&envId=top-100-liked "1. 两数之和 - 力扣（LeetCode）")

```bash
给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。
```

```c++
class Solution {
public:
    // 暴力搜索
    vector<int> twoSum2(vector<int>& nums, int target) {
        int size = nums.size();

        for(int i = 0; i < size; i++)
        {
            for (int j = i + 1; j < size; j++)
            {
                if (nums[i] + nums[j] == target)
                    return {i, j};
            }
        }

        return {};
    }

    // 哈希表
    // 创建一个哈希表，对于每一个 x，我们首先查询哈希表中是否存在 target - x，
    // 然后将 x 插入到哈希表中，即可保证不会让 x 和自己匹配。
    vector<int> twoSum(vector<int>& nums, int target) {
        std::unordered_map<int, int> hash_table;

        for (int i = 0; i < nums.size(); i++)
        {
            auto it = hash_table.find(target - nums[i]);
            if (it != hash_table.end())
            {
                return {i, it->second};
            }
            hash_table[nums[i]] = i;
        }

        return {};
    }
};
```

# 2.字母异位词分组

[49. 字母异位词分组 - 力扣（LeetCode）](https://leetcode.cn/problems/group-anagrams/description/ "49. 字母异位词分组 - 力扣（LeetCode）")

```bash
给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。

字母异位词 是由重新排列源单词的所有字母得到的一个新单词。
```

1.  排序，匹配：对两个字符串，排序之后得到的字符串一定是相同的，可以将排序后的字符串作为哈希表的键
2.  hash 计数 ： 两个字符串中相同字母出现的次数一样，可以将每个字母出现的次数使用字符串表示，作为hash的键；由于字符串只包含小写，使用长度为26的数组记录每个字母的出现次数。

```c++
// 1.排序，匹配
// 2.hash 计数
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
```

# 3.最长连续序列

[128. 最长连续序列 - 力扣（LeetCode）](https://leetcode.cn/problems/longest-consecutive-sequence/description/?envType=study-plan-v2\&envId=top-100-liked "128. 最长连续序列 - 力扣（LeetCode）")

```bash
给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 O(n) 的算法解决此问题。
```

1.  排序，然后遍历数组。和上一项差值为1或0，则连续，否则为不连续（时间快）
2.  哈希表 + 无序集合（去重）。如果集合中不存在 nums\[i - 1]，则开始从判断 nums\[i + 1...n]的值是否在集合内部

```c++
class Solution {
public:
    // 1.排序 + 遍历
    int longestConsecutive1(vector<int>& nums) {
        int max_len = 0;
        if (nums.size() == 0) {
            return 0;
        } else if (nums.size() == 1) {
            return 1;
        }

        // 排序
        std::sort(nums.begin(), nums.end());

        int tmp = 0;
        for (int i = 1; i < nums.size(); i++) {
            int diff = nums[i] - nums[i - 1];
            // diff == 0, 有相同的数字，跳过，连续值不变
            if (diff == 0) {
                // tmp += 0;
                continue;
            } else if (diff == 1) { // diff=1，连续，连续值加1
                tmp += 1;
            } else {    // 其他不是连续，将连续值置为 0 
                tmp = 0;
            }

            max_len = std::max(max_len, tmp);
        }

        // 因为从第 2 个数组开始遍历的
        return max_len + 1;
    }

    // 2.哈希表 + 无序集合（去重）
    int longestConsecutive(vector<int>& nums) {
        // 集合去重
        std::unordered_set<int> num_set;
        for (const int& num : nums) {
            num_set.insert(num);
        }

        int max_len = 0;

        // 遍历集合
        for (const int& num : num_set) {
            // 如果集合中不存在 nums[i - 1]，则开始从判断 nums[i + 1...n]的值是否在集合内部
            if (!num_set.count(num - 1)) {
                // 初始化从nums[i]开始的数值和长度
                int curr_num = num;
                int curr_len = 1;
                // 依次判断是否存在下一个连续的
                while (num_set.count(curr_num + 1))
                {
                    curr_num += 1;
                    curr_len += 1;
                }
                // 更新最大值
                max_len = std::max(max_len, curr_len);
            }
        }

        return max_len;
    }
    
};
```
