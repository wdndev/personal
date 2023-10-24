/*
 * @lc app=leetcode.cn id=66 lang=cpp
 *
 * [66] 加一
 *
 * https://leetcode.cn/problems/plus-one/description/
 *
 * algorithms
 * Easy (45.23%)
 * Likes:    1312
 * Dislikes: 0
 * Total Accepted:    675.9K
 * Total Submissions: 1.5M
 * Testcase Example:  '[1,2,3]'
 *
 * 给定一个由 整数 组成的 非空 数组所表示的非负整数，在该数的基础上加一。
 * 
 * 最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
 * 
 * 你可以假设除了整数 0 之外，这个整数不会以零开头。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：digits = [1,2,3]
 * 输出：[1,2,4]
 * 解释：输入数组表示数字 123。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：digits = [4,3,2,1]
 * 输出：[4,3,2,2]
 * 解释：输入数组表示数字 4321。
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：digits = [0]
 * 输出：[1]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 
 * 0 
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 只需要对数组 digits进行一次逆序遍历，找出第一个不为 999 的元素，
    // 将其加一并将后续所有元素置零即可。如果 digits中所有的元素均为 999，
    // 那么对应着「思路」部分的第三种情况，我们需要返回一个新的数组。

    vector<int> plusOne(vector<int>& digits) {
        int n = digits.size();
        for (int i = n - 1; i >= 0; i--)
        {
            if (digits[i] != 9)
            {
                digits[i]++;
                for ( int j = i + 1; j < n; j++)
                {
                    digits[j] = 0;
                }
                return digits;
            }
        }
        
        // 所有元素均为9
        std::vector<int> res(n + 1);
        res[0] = 1;
        return res;
    }
};
// @lc code=end

