/*
 * @lc app=leetcode.cn id=2960 lang=cpp
 *
 * [2960] 统计已测试设备
 *
 * https://leetcode.cn/problems/count-tested-devices-after-test-operations/description/
 *
 * algorithms
 * Easy (83.54%)
 * Likes:    24
 * Dislikes: 0
 * Total Accepted:    26.2K
 * Total Submissions: 29.7K
 * Testcase Example:  '[1,1,2,1,3]'
 *
 * 给你一个长度为 n 、下标从 0 开始的整数数组 batteryPercentages ，表示 n 个设备的电池百分比。
 * 
 * 你的任务是按照顺序测试每个设备 i，执行以下测试操作：
 * 
 * 
 * 如果 batteryPercentages[i] 大于 0：
 * 
 * 
 * 增加 已测试设备的计数。
 * 将下标在 [i + 1, n - 1] 的所有设备的电池百分比减少 1，确保它们的电池百分比 不会低于 0 ，即
 * batteryPercentages[j] = max(0, batteryPercentages[j] - 1)。
 * 移动到下一个设备。
 * 
 * 
 * 否则，移动到下一个设备而不执行任何测试。
 * 
 * 
 * 返回一个整数，表示按顺序执行测试操作后 已测试设备 的数量。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：batteryPercentages = [1,1,2,1,3]
 * 输出：3
 * 解释：按顺序从设备 0 开始执行测试操作：
 * 在设备 0 上，batteryPercentages[0] > 0 ，现在有 1 个已测试设备，batteryPercentages 变为
 * [1,0,1,0,2] 。
 * 在设备 1 上，batteryPercentages[1] == 0 ，移动到下一个设备而不进行测试。
 * 在设备 2 上，batteryPercentages[2] > 0 ，现在有 2 个已测试设备，batteryPercentages 变为
 * [1,0,1,0,1] 。
 * 在设备 3 上，batteryPercentages[3] == 0 ，移动到下一个设备而不进行测试。
 * 在设备 4 上，batteryPercentages[4] > 0 ，现在有 3 个已测试设备，batteryPercentages 保持不变。
 * 因此，答案是 3 。
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：batteryPercentages = [0,1,2]
 * 输出：2
 * 解释：按顺序从设备 0 开始执行测试操作：
 * 在设备 0 上，batteryPercentages[0] == 0 ，移动到下一个设备而不进行测试。
 * 在设备 1 上，batteryPercentages[1] > 0 ，现在有 1 个已测试设备，batteryPercentages 变为
 * [0,1,1] 。
 * 在设备 2 上，batteryPercentages[2] > 0 ，现在有 2 个已测试设备，batteryPercentages 保持不变。
 * 因此，答案是 2 。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= n == batteryPercentages.length <= 100 
 * 0 <= batteryPercentages[i] <= 100
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    int countTestedDevices(vector<int>& batteryPercentages) {
        int ans = 0;
        for (int i = 0; i < batteryPercentages.size(); i++) {
            int left = i - 1;
            int right = i + 1;
            if (batteryPercentages[i] > 0) {
                for (int j = i + 1; j < batteryPercentages.size(); j++) {
                    batteryPercentages[j] = batteryPercentages[j] - 1;
                }
                ans++;
            } 
        }

        return ans;
    }
};
// @lc code=end

