/*
 * @lc app=leetcode.cn id=312 lang=cpp
 *
 * [312] 戳气球
 *
 * https://leetcode.cn/problems/burst-balloons/description/
 *
 * algorithms
 * Hard (69.94%)
 * Likes:    1383
 * Dislikes: 0
 * Total Accepted:    126.6K
 * Total Submissions: 179.2K
 * Testcase Example:  '[3,1,5,8]'
 *
 * 有 n 个气球，编号为0 到 n - 1，每个气球上都标有一个数字，这些数字存在数组 nums 中。
 * 
 * 现在要求你戳破所有的气球。戳破第 i 个气球，你可以获得 nums[i - 1] * nums[i] * nums[i + 1] 枚硬币。 这里的 i
 * - 1 和 i + 1 代表和 i 相邻的两个气球的序号。如果 i - 1或 i + 1 超出了数组的边界，那么就当它是一个数字为 1 的气球。
 * 
 * 求所能获得硬币的最大数量。
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [3,1,5,8]
 * 输出：167
 * 解释：
 * nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
 * coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 167
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [1,5]
 * 输出：10
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * n == nums.length
 * 1 <= n <= 300
 * 0 <= nums[i] <= 100
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 假设这个区间是个开区间，最左边索引 i，最右边索引 j, 只能戳爆 i 和 j 之间的气球，i 和 j 不要戳
    // dp[i][j] 表示开区间 (i,j) 内你能拿到的最多金币
    // 在 (i,j) 开区间得到的金币可以由 dp[i][k] 和 dp[k][j] 进行转移
    // 此刻选择戳爆气球 k，那么得到的金币数量就是：
    // dp[i][j] = dp[i][k] + nums_[i] * nums_[k] * nums[j] + dp[k][j]
    // val[i] 表示 i 位置气球的数字, 然后 (i,k) 和 (k,j) 也都是开区间
    // 为什么前后只要加上 dp[i][k] 和 dp[k][j] 的值?
    // 因为 k 是最后一个被戳爆的，所以 (i,j) 区间中 k 两边的东西必然是先各自被戳爆了的，
    // 所以把 (i,k) 开区间所有气球戳爆，然后把戳爆这些气球的所有金币都收入囊中，金币数量记录在 dp[i][k]
    int maxCoins(vector<int>& nums) {
        int n = nums.size();
        m_nums.resize(n + 2, 1);
        m_dp.resize(n + 2, std::vector<int>(n + 2, 0));

        for (int i =  0; i < n; i++) {
            m_nums[i + 1] = nums[i];
        }

        // 保证计算状态顺序 从区间长度枚举
        for (int len = 3; len < n + 3; len++) {
            // 再枚举起始点
            for (int i = 0; i + len < n + 3; i++) {
                this->helper(i, i + len - 1);
            }
        }

        return m_dp[0][n + 1];
    }
private:
    std::vector<int> m_nums;
    std::vector<std::vector<int>> m_dp;

    // 计算 开区间 (l, r)内 的答案
    // 枚举最后一个戳破的气球k 则转移方程为
    // dp[i][j] = dp[i][k] + nums_[i] * nums_[k] * nums[j] + dp[k][j]
    // k的取值范围是 i + 1 ~ j - 1
    void helper(int l, int r) {
        int ans = 0;
        for (int k = l + 1; k < r; k++) {
            int left = m_dp[l][k];
            int right = m_dp[k][r];

            ans = std::max(ans, left + m_nums[l] * m_nums[k] * m_nums[r] + right);
        }

        m_dp[l][r] = ans;
    }
};
// @lc code=end

