/*
 * @lc app=leetcode.cn id=204 lang=cpp
 *
 * [204] 计数质数
 *
 * https://leetcode.cn/problems/count-primes/description/
 *
 * algorithms
 * Medium (37.08%)
 * Likes:    1116
 * Dislikes: 0
 * Total Accepted:    265.4K
 * Total Submissions: 715.8K
 * Testcase Example:  '10'
 *
 * 给定整数 n ，返回 所有小于非负整数 n 的质数的数量 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：n = 10
 * 输出：4
 * 解释：小于 10 的质数一共有 4 个, 它们是 2, 3, 5, 7 。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：n = 0
 * 输出：0
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：n = 1
 * 输出：0
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 0 <= n <= 5 * 10^6
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 1.枚举, 超时
    bool is_prime(int x) {
        for (int i = 2; i <= pow(x, 0.5); i++) {
            if (x % i == 0) {
                return false;
            }
        }
        return true;
    }
    int countPrimes1(int n) {
        int count = 0;
        for (int i = 2; i < n; i++) {
            if (this->is_prime(i)) {
                count++;
            }
        }
        return count;
    }

    // 2.埃氏筛法
    int countPrimes(int n) {
        std::vector<bool> prime_flag(n, true);
        int count = 0;

        for (int i = 2; i < n; i++) {
            if (prime_flag[i]) {
                count++;
                if ((long long)i * i < n) {
                    for (int j = i * i; j < n; j = j + i) {
                        prime_flag[j] = false;
                    }
                }
                
            }
        }

        return count;
    }

};
// @lc code=end

