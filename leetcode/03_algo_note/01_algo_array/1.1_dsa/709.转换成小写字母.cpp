/*
 * @lc app=leetcode.cn id=709 lang=cpp
 *
 * [709] 转换成小写字母
 *
 * https://leetcode.cn/problems/to-lower-case/description/
 *
 * algorithms
 * Easy (76.43%)
 * Likes:    235
 * Dislikes: 0
 * Total Accepted:    134.7K
 * Total Submissions: 176.1K
 * Testcase Example:  '"Hello"'
 *
 * 给你一个字符串 s ，将该字符串中的大写字母转换成相同的小写字母，返回新的字符串。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：s = "Hello"
 * 输出："hello"
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：s = "here"
 * 输出："here"
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：s = "LOVELY"
 * 输出："lovely"
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 
 * s 由 ASCII 字符集中的可打印字符组成
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    string toLowerCase(string s) {
        
        for (int i = 0; i < s.size(); i++) {
            if (s[i] >= 'A' && s[i] <= 'Z') {
                s[i] = s[i] + 32;
            }
        }

        return s;
    }
};
// @lc code=end

