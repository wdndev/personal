/*
 * @lc app=leetcode.cn id=338 lang=cpp
 *
 * [338] 比特位计数
 *
 * https://leetcode.cn/problems/counting-bits/description/
 *
 * algorithms
 * Easy (78.64%)
 * Likes:    1268
 * Dislikes: 0
 * Total Accepted:    314.1K
 * Total Submissions: 399.4K
 * Testcase Example:  '2'
 *
 * 给你一个整数 n ，对于 0 <= i <= n 中的每个 i ，计算其二进制表示中 1 的个数 ，返回一个长度为 n + 1 的数组 ans
 * 作为答案。
 * 
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：n = 2
 * 输出：[0,1,1]
 * 解释：
 * 0 --> 0
 * 1 --> 1
 * 2 --> 10
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：n = 5
 * 输出：[0,1,1,2,1,2]
 * 解释：
 * 0 --> 0
 * 1 --> 1
 * 2 --> 10
 * 3 --> 11
 * 4 --> 100
 * 5 --> 101
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 0 <= n <= 10^5
 * 
 * 
 * 
 * 
 * 进阶：
 * 
 * 
 * 很容易就能实现时间复杂度为 O(n log n) 的解决方案，你可以在线性时间复杂度 O(n) 内用一趟扫描解决此问题吗？
 * 你能不使用任何内置函数解决此问题吗？（如，C++ 中的 __builtin_popcount ）
 * 
 * 
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.求位数比较
    // 对于任意整数 xxx，令 x=x & (x−1)，该运算将 x 的二进制表示的最后一个 1 变成 0。
    // 因此，对 x 重复该操作，直到 x 变成 0，则操作次数即为 x 的「一比特数」。
    int countOnes(int x) {
        int ones = 0;
        while (x > 0) {
            x &= (x - 1);
            ones++;
        }
        return ones;
    }

    vector<int> countBits(int n) {
        vector<int> bits(n + 1);
        for (int i = 0; i <= n; i++) {
            bits[i] = countOnes(i);
        }
        return bits;
    }
    

    // 2.动态规划——最低设置位
    vector<int> countBits2(int n) {
        std::vector<int> bits(n + 1, 0);
        for (int i = 1; i <= n; i++) {
            bits[i] += bits[i & (i - 1)] + 1;
        }

        return bits;
    }
};
// @lc code=end

