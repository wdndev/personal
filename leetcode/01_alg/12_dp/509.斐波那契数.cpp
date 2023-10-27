/*
 * @lc app=leetcode.cn id=509 lang=cpp
 *
 * [509] 斐波那契数
 *
 * https://leetcode.cn/problems/fibonacci-number/description/
 *
 * algorithms
 * Easy (66.00%)
 * Likes:    710
 * Dislikes: 0
 * Total Accepted:    614.3K
 * Total Submissions: 931K
 * Testcase Example:  '2'
 *
 * 斐波那契数 （通常用 F(n) 表示）形成的序列称为 斐波那契数列 。该数列由 0 和 1 开始，后面的每一项数字都是前面两项数字的和。也就是：
 * 
 * 
 * F(0) = 0，F(1) = 1
 * F(n) = F(n - 1) + F(n - 2)，其中 n > 1
 * 
 * 
 * 给定 n ，请计算 F(n) 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：n = 2
 * 输出：1
 * 解释：F(2) = F(1) + F(0) = 1 + 0 = 1
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：n = 3
 * 输出：2
 * 解释：F(3) = F(2) + F(1) = 1 + 1 = 2
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：n = 4
 * 输出：3
 * 解释：F(4) = F(3) + F(2) = 2 + 1 = 3
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 0 <= n <= 30
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.递归
    int fib1(int n) {
        return n <= 1 ? n : this->fib(n - 1) + this->fib(n - 2);
    }

    // 2.递归 + 记忆化搜索
    int fib(int n) {
        if (n <= 1) {
            return n;
        }

        std::vector<int> memo(n+1, 0);
        return this->fib2(n, memo);
    }
    int fib2(int n, std::vector<int>& memo) {
        if (n <= 1) {
            return n;
        }

        if (memo[n] == 0) {
            memo[n] = this->fib2(n - 1, memo) + fib2(n -2, memo);
        }

        return memo[n];
    }

    // 3.自顶向下
    int fib3(int n) {
        if (n <= 1) {
            return n;
        }

        std::vector<int> memo(n+1, 0);
        memo[0] = 0;
        memo[1] = 1;
        for (int i = 2; i <= n; i++) {
            memo[i] = memo[i - 2] + memo[i - 1];
        }

        return memo[n];
    }
};
// @lc code=end

