/*
 * @lc app=leetcode.cn id=15 lang=cpp
 *
 * [15] 三数之和
 *
 * https://leetcode.cn/problems/3sum/description/
 *
 * algorithms
 * Medium (37.41%)
 * Likes:    6455
 * Dislikes: 0
 * Total Accepted:    1.5M
 * Total Submissions: 4.1M
 * Testcase Example:  '[-1,0,1,2,-1,-4]'
 *
 * 给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j !=
 * k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请
 * 
 * 你返回所有和为 0 且不重复的三元组。
 * 
 * 注意：答案中不可以包含重复的三元组。
 * 
 * 
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [-1,0,1,2,-1,-4]
 * 输出：[[-1,-1,2],[-1,0,1]]
 * 解释：
 * nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
 * nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
 * nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0 。
 * 不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
 * 注意，输出的顺序和三元组的顺序并不重要。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [0,1,1]
 * 输出：[]
 * 解释：唯一可能的三元组和不为 0 。
 * 
 * 
 * 示例 3：
 * 
 * 
 * 输入：nums = [0,0,0]
 * 输出：[[0,0,0]]
 * 解释：唯一可能的三元组和为 0 。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 3 <= nums.length <= 3000
 * -10^5 <= nums[i] <= 10^5
 * 
 * 
 */

// @lc code=start
// a + b = -c

// 1.暴力求解，需要三重循环 O(n^3)
// 2.hash表来记录， a+b到hash表中去查，是否存在-c
// 3.左右下标推进
class Solution {
public:
    // 1.暴力求解方法
    vector<vector<int>> threeSum1(vector<int>& nums) {
        std::vector<std::vector<int>> ans;
        for (int i = 0; i < nums.size() - 2; i++) {
            for (int j = i + 1; j < nums.size() - 1; j++) {
                for (int k = j + 1; k < nums.size(); k++) {
                    if (nums[i] + nums[j] + nums[k] == 0) {
                        ans.push_back({nums[i], nums[j], nums[k]});
                    }
                }
            }
        }

        return ans;
    }

    // 3.左右下标推进
    // 固定3个指针中最左数字的指针k，双指针i，j分设在数据两端
    // 通过双指针交替向中间移动，记录对于每个固定指针k所有满足
    // nums[k] + nums[i] + nums[j] == 0 的 i, j 组合
    vector<vector<int>> threeSum(vector<int>& nums) {
        int size = nums.size();
        if (size < 3)
            return {};
        std::vector<std::vector<int>> res;
        // 排序
        std::sort(nums.begin(), nums.end());
        // 固定第一个数，转化为求两数之和
        for(int i = 0; i < size; i++)
        {
            // 如果第一个数为正数，因为是递增的，后面你的数不可能为0了
            if (nums[i] > 0)
                return res;
            // 去重，如果被选过了，跳过
            if (i > 0 && nums[i] == nums[i-1])
                continue;
            // 双指针在nums[i]后面的区间中寻找和为0-nums[i]的另外两个数
            int left = i + 1;
            int right = size - 1;
            while(left < right)
            {
                // 两数之和太大，右指针左移
                if (nums[left] + nums[right] > -nums[i])
                    right--;
                // 两数之和太小，左指针右移
                else if(nums[left] + nums[right] < -nums[i])
                    left++;
                else
                {
                    // 找到一个和为零的三元组，添加到结果中，左右指针内缩，继续寻找
                    res.push_back(std::vector<int>{nums[i], nums[left], nums[right]});
                    left++;
                    right--;

                    // 去重：第二个数和第三个数也不重复选取
                    // 例如：[-4,1,1,1,2,3,3,3], i=0, left=1, right=5
                    while (left < right && nums[left] == nums[left-1])  left++;
                    while (left < right && nums[right] == nums[right+1])    right--;
                }
            }
        }

        return res;
    }
};
// @lc code=end

