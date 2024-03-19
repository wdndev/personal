/*
 * @lc app=leetcode.cn id=303 lang=cpp
 *
 * [303] 区域和检索 - 数组不可变
 *
 * https://leetcode.cn/problems/range-sum-query-immutable/description/
 *
 * algorithms
 * Easy (78.25%)
 * Likes:    624
 * Dislikes: 0
 * Total Accepted:    262.5K
 * Total Submissions: 335.6K
 * Testcase Example:  '["NumArray","sumRange","sumRange","sumRange"]\n' +
  '[[[-2,0,3,-5,2,-1]],[0,2],[2,5],[0,5]]'
 *
 * 给定一个整数数组  nums，处理以下类型的多个查询:
 * 
 * 
 * 计算索引 left 和 right （包含 left 和 right）之间的 nums 元素的 和 ，其中 left <= right
 * 
 * 
 * 实现 NumArray 类：
 * 
 * 
 * NumArray(int[] nums) 使用数组 nums 初始化对象
 * int sumRange(int i, int j) 返回数组 nums 中索引 left 和 right 之间的元素的 总和 ，包含 left 和
 * right 两点（也就是 nums[left] + nums[left + 1] + ... + nums[right] )
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：
 * ["NumArray", "sumRange", "sumRange", "sumRange"]
 * [[[-2, 0, 3, -5, 2, -1]], [0, 2], [2, 5], [0, 5]]
 * 输出：
 * [null, 1, -1, -3]
 * 
 * 解释：
 * NumArray numArray = new NumArray([-2, 0, 3, -5, 2, -1]);
 * numArray.sumRange(0, 2); // return 1 ((-2) + 0 + 3)
 * numArray.sumRange(2, 5); // return -1 (3 + (-5) + 2 + (-1)) 
 * numArray.sumRange(0, 5); // return -3 ((-2) + 0 + 3 + (-5) + 2 + (-1))
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= nums.length <= 10^4
 * -10^5 <= nums[i] <= 10^5
 * 0 <= i <= j < nums.length
 * 最多调用 10^4 次 sumRange 方法
 * 
 * 
 */

// @lc code=start
class NumArray {
public:
    NumArray(vector<int>& nums) {
        m_nums = nums;
    }
    
    int sumRange(int left, int right) {
        int sum = 0;
        for (int i = left; i <= right; i++) {
            sum += m_nums[i];
        }
        return sum;
    }
private:
    std::vector<int> m_nums;
};

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray* obj = new NumArray(nums);
 * int param_1 = obj->sumRange(left,right);
 */
// @lc code=end

