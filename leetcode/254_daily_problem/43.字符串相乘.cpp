/*
 * @lc app=leetcode.cn id=43 lang=cpp
 *
 * [43] 字符串相乘
 *
 * https://leetcode.cn/problems/multiply-strings/description/
 *
 * algorithms
 * Medium (44.31%)
 * Likes:    1332
 * Dislikes: 0
 * Total Accepted:    339.8K
 * Total Submissions: 766.9K
 * Testcase Example:  '"2"\n"3"'
 *
 * 给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。
 * 
 * 注意：不能使用任何内置的 BigInteger 库或直接将输入转换为整数。
 * 
 * 
 * 
 * 示例 1:
 * 
 * 
 * 输入: num1 = "2", num2 = "3"
 * 输出: "6"
 * 
 * 示例 2:
 * 
 * 
 * 输入: num1 = "123", num2 = "456"
 * 输出: "56088"
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= num1.length, num2.length <= 200
 * num1 和 num2 只能由数字组成。
 * num1 和 num2 都不包含任何前导零，除了数字0本身。
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    string multiply(string num1, string num2) {
        int m = num1.size();
        int n = num2.size();
        // 结果最多为 m+n 位数
        std::vector<int> ans(m + n, 0);
        // 从个位数开始逐位相乘
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                int mul = (num1[i] - '0') * (num2[j] - '0');
                // 乘积在 ans 中对应索引位置
                int p1 = i + j;
                int p2 = i + j + 1;
                // 叠加到 ans 中
                int sum = mul + ans[p2];
                ans[p2] = sum % 10;
                ans[p1] += sum / 10;
            }
        }
        // for (auto x : ans) 
        //     std::cout << x << std::endl;
        // 结果前可能存在好多 0
        int i = 0;
        while (i < ans.size() && ans[i] == 0) {
            i++;
        }
        // 将计算结果转化位字符串
        std::string str;
        for (; i < ans.size(); i++) {
            str.push_back('0' + ans[i]);
        }

        return str.size() == 0 ? "0" : str;
    }
};
// @lc code=end

