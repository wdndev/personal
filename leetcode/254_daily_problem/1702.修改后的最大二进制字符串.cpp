/*
 * @lc app=leetcode.cn id=1702 lang=cpp
 *
 * [1702] 修改后的最大二进制字符串
 *
 * https://leetcode.cn/problems/maximum-binary-string-after-change/description/
 *
 * algorithms
 * Medium (48.59%)
 * Likes:    59
 * Dislikes: 0
 * Total Accepted:    9.3K
 * Total Submissions: 17.9K
 * Testcase Example:  '"000110"'
 *
 * 给你一个二进制字符串 binary ，它仅有 0 或者 1 组成。你可以使用下面的操作任意次对它进行修改：
 * 
 * 
 * 操作 1 ：如果二进制串包含子字符串 "00" ，你可以用 "10" 将其替换。
 * 
 * 
 * 比方说， "00010" -> "10010"
 * 
 * 
 * 操作 2 ：如果二进制串包含子字符串 "10" ，你可以用 "01" 将其替换。
 * 
 * 比方说， "00010" -> "00001"
 * 
 * 
 * 
 * 
 * 请你返回执行上述操作任意次以后能得到的 最大二进制字符串 。如果二进制字符串 x 对应的十进制数字大于二进制字符串 y
 * 对应的十进制数字，那么我们称二进制字符串 x 大于二进制字符串 y 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：binary = "000110"
 * 输出："111011"
 * 解释：一个可行的转换为：
 * "000110" -> "000101" 
 * "000101" -> "100101" 
 * "100101" -> "110101" 
 * "110101" -> "110011" 
 * "110011" -> "111011"
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：binary = "01"
 * 输出："01"
 * 解释："01" 没办法进行任何转换。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 
 * binary 仅包含 '0' 和 '1' 。
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 操作1 ： 00 -> 10
    // 操作2 :  10 -> 01
    // 提示1 ： 答案中不会包含 00 （连续的0）。如果有 00 会变成更大的 10
    // 提示2 ： 答案中最多包含一个 0。可以通过操作2，让最高位0的右侧也变成0，然后通过错做1，使00 -> 10
    // 提示3 ： 如果全为 1， 则返回
    // 设 binary 从左到右的第一个0的下表为i，为了得到更大的二进制数，下标在 [i, n-1]中的1会随着0的左移被挤到binary末尾。
    // 一般的，设[i, n-1]中有 cnt_1个1，那么答案中唯一的0的下标为 n-1-cnt_1 
    string maximumBinaryString(string binary) {
        int i = binary.find('0');
        // 如果全是1，则返回
        if (i < 0) {
            return binary;
        }
        // 统计 [i, n-1] 中 '1' 的个数
        int cnt1 = count(binary.begin() + i, binary.end(), '1');
        return std::string(binary.size() - 1 - cnt1, '1') + '0' + std::string(cnt1, '1');
    }
};
// @lc code=end

