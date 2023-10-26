/*
 * @lc app=leetcode.cn id=433 lang=cpp
 *
 * [433] 最小基因变化
 *
 * https://leetcode.cn/problems/minimum-genetic-mutation/description/
 *
 * algorithms
 * Medium (54.75%)
 * Likes:    266
 * Dislikes: 0
 * Total Accepted:    55.7K
 * Total Submissions: 102K
 * Testcase Example:  '"AACCGGTT"\n"AACCGGTA"\n["AACCGGTA"]'
 *
 * 基因序列可以表示为一条由 8 个字符组成的字符串，其中每个字符都是 'A'、'C'、'G' 和 'T' 之一。
 * 
 * 假设我们需要调查从基因序列 start 变为 end 所发生的基因变化。一次基因变化就意味着这个基因序列中的一个字符发生了变化。
 * 
 * 
 * 例如，"AACCGGTT" --> "AACCGGTA" 就是一次基因变化。
 * 
 * 
 * 另有一个基因库 bank 记录了所有有效的基因变化，只有基因库中的基因才是有效的基因序列。（变化后的基因必须位于基因库 bank 中）
 * 
 * 给你两个基因序列 start 和 end ，以及一个基因库 bank ，请你找出并返回能够使 start 变化为 end
 * 所需的最少变化次数。如果无法完成此基因变化，返回 -1 。
 * 
 * 注意：起始基因序列 start 默认是有效的，但是它并不一定会出现在基因库中。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：start = "AACCGGTT", end = "AACCGGTA", bank = ["AACCGGTA"]
 * 输出：1
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：start = "AACCGGTT", end = "AAACGGTA", bank =
 * ["AACCGGTA","AACCGCTA","AAACGGTA"]
 * 输出：2
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：start = "AAAAACCC", end = "AACCCCCC", bank =
 * ["AAAACCCC","AAACCCCC","AACCCCCC"]
 * 输出：3
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * start.length == 8
 * end.length == 8
 * 0 <= bank.length <= 10
 * bank[i].length == 8
 * start、end 和 bank[i] 仅由字符 ['A', 'C', 'G', 'T'] 组成
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // BFS
    // 经过分析可知，题目要求将一个基因序列 A 变化至另一个基因序列 B，需要满足以下条件：
    // 1.序列 A 与 序列 B 之间只有一个字符不同；
    // 2.变化字符只能从 ‘A’, ‘C’, ‘G’, ‘T’中进行选择；
    // 3.变换后的序列 B 一定要在字符串数组 bank中。

    // 步骤如下：
    // 1.如果 start 与 end相等，此时直接返回 0；如果最终的基因序列不在 bank中，则此时按照题意要求，无法生成，直接返回 −1；
    // 2.首先我们将可能变换的基因 s 从队列中取出，按照上述的变换规则，尝试所有可能的变化后的基因，
    //   比如一个 AACCGGTA，我们依次尝试改变基因 s 的一个字符，并尝试所有可能的基因变化序列 s0,s1,s2,⋯ ,si,⋯ ,s23，
    //   变化一次最多可能会生成 3×8=24 种不同的基因序列。
    // 3.需要检测当前生成的基因序列的合法性 si，首先利用哈希表检测 si 是否在数组 bank 中，
    //   如果是则认为该基因合法，否则改变化非法直接丢弃；其次还需要用哈希表记录已经遍历过的基因序列，
    //   如果该基因序列已经遍历过，则此时直接跳过；如果合法且未遍历过的基因序列，则将其加入到队列中。
    // 4.如果当前变换后的基因序列与 end 相等，则此时我们直接返回最小的变化次数即可；
    //   如果队列中所有的元素都已经遍历完成还无法变成 end，则此时无法实现目标变化，返回 −1。
    int minMutation(string startGene, string endGene, vector<string>& bank) {
        std::unordered_set<std::string> cnt;
        std::unordered_set<std::string> visited;
        char keys[4] = {'A', 'C', 'G', 'T'};

        for (auto& b : bank) {
            cnt.emplace(b);
        }

        if (startGene == endGene) {
            return 0;
        }

        if (!cnt.count(endGene)) {
            return -1;
        }

        std::queue<std::string> queue;
        queue.emplace(startGene);
        int step = 1;

        while (!queue.empty()) {
            int size = queue.size();

            for (int i = 0; i < size; i++) {
                std::string curr = queue.front();
                queue.pop();

                for (int j = 0; j < 8; j++) {
                    for (int k = 0; k < 4; k++) {
                        if (keys[k] != curr[j]) {
                            std::string next = curr;
                            next[j] = keys[k];

                            if (!visited.count(next) && cnt.count(next)) {
                                if (next == endGene) {
                                    return step;
                                }
                                queue.emplace(next);
                                visited.emplace(next);
                            }
                        }
                    }
                }
            }

            step++;
        }

        return -1;
    }
};
// @lc code=end

