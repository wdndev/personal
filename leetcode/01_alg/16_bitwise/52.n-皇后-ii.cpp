/*
 * @lc app=leetcode.cn id=52 lang=cpp
 *
 * [52] N 皇后 II
 *
 * https://leetcode.cn/problems/n-queens-ii/description/
 *
 * algorithms
 * Hard (82.36%)
 * Likes:    480
 * Dislikes: 0
 * Total Accepted:    131.7K
 * Total Submissions: 160K
 * Testcase Example:  '4'
 *
 * n 皇后问题 研究的是如何将 n 个皇后放置在 n × n 的棋盘上，并且使皇后彼此之间不能相互攻击。
 * 
 * 给你一个整数 n ，返回 n 皇后问题 不同的解决方案的数量。
 * 
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：n = 4
 * 输出：2
 * 解释：如上图所示，4 皇后问题存在两个不同的解法。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：n = 1
 * 输出：1
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= n <= 9
 * 
 * 
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    int totalNQueens(int n) {
        m_count = 0;
        m_size = (1 << n) - 1;
        this->dfs(0, 0, 0);
        return m_count;
    }
    
private:
    int m_size;
    int m_count;

    void dfs(int row, int pie, int na) {
        // 递归终止条件
        if (row == m_size) {
            m_count++;
            return;
        }
        // 得到当前所有空位
        int pos = m_size & (~(row | pie | na));

        while (pos != 0)
        {
            // 取到最低位的1
            int p = pos & (-pos);
            // 将p位置放入皇后
            pos -= p; // pos &= pos - 1
            this->dfs(row | p, (pie | p) << 1, (na | p) >> 1);
            // 不需要revert cols, pie, na 的状态
        }
        
    }
};
// @lc code=end

