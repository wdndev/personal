/*
 * @lc app=leetcode.cn id=1235 lang=cpp
 *
 * [1235] 规划兼职工作
 *
 * https://leetcode.cn/problems/maximum-profit-in-job-scheduling/description/
 *
 * algorithms
 * Hard (57.55%)
 * Likes:    439
 * Dislikes: 0
 * Total Accepted:    39.8K
 * Total Submissions: 67.5K
 * Testcase Example:  '[1,2,3,3]\n[3,4,5,6]\n[50,10,40,70]'
 *
 * 你打算利用空闲时间来做兼职工作赚些零花钱。
 * 
 * 这里有 n 份兼职工作，每份工作预计从 startTime[i] 开始到 endTime[i] 结束，报酬为 profit[i]。
 * 
 * 给你一份兼职工作表，包含开始时间 startTime，结束时间 endTime 和预计报酬 profit 三个数组，请你计算并返回可以获得的最大报酬。
 * 
 * 注意，时间上出现重叠的 2 份工作不能同时进行。
 * 
 * 如果你选择的工作在时间 X 结束，那么你可以立刻进行在时间 X 开始的下一份工作。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 
 * 输入：startTime = [1,2,3,3], endTime = [3,4,5,6], profit = [50,10,40,70]
 * 输出：120
 * 解释：
 * 我们选出第 1 份和第 4 份工作， 
 * 时间范围是 [1-3]+[3-6]，共获得报酬 120 = 50 + 70。
 * 
 * 
 * 示例 2：
 * 
 * ⁠
 * 
 * 输入：startTime = [1,2,3,4,6], endTime = [3,5,10,6,9], profit =
 * [20,20,100,70,60]
 * 输出：150
 * 解释：
 * 我们选择第 1，4，5 份工作。 
 * 共获得报酬 150 = 20 + 70 + 60。
 * 
 * 
 * 示例 3：
 * 
 * 
 * 
 * 输入：startTime = [1,1,1], endTime = [2,3,4], profit = [5,6,4]
 * 输出：6
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= startTime.length == endTime.length == profit.length <= 5 * 10^4
 * 1 <= startTime[i] < endTime[i] <= 10^9
 * 1 <= profit[i] <= 10^4
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    // 动态规划 + 二分查找优化
    // 1.将工作按照结束时间排序
    // 2.分类讨论，求出按照结束时间排序的前i个工作的最大报酬，f[i]
    //  - 不选第i个工作：f[i] = f[i - 1]
    //  - 选第i个工作： f[i]=f[j] + profit[i]，其中j是最大的满足endtime[j] < starttime[i]的j，不存在时为-1
    //  取两者的最大值，即 f[i] = max(f[i - 1], f[j] + profile]i)
    // 3.由于i=0时，i-1会变成-1，所以改写为：f[i+1]=max(f[i], f[j+1]+profit[i])
    // 4.初始为 f[0]=0
    int jobScheduling(vector<int>& startTime, vector<int>& endTime, vector<int>& profit) {
        int n = startTime.size();
        std::vector<std::vector<int>> jobs(n, std::vector<int>(3));
        for (int i = 0; i < n; i++) {
            jobs[i] = {startTime[i], endTime[i], profit[i]};
        }
        // 按照结束时间排序
        std::sort(jobs.begin(), jobs.end(), [](auto& a, auto& b){
            return a[1] < b[1];
        });

        std::vector<int> f(n + 1);
        for (int i = 0; i < n; i++) {
            int j = this->search(jobs, i, jobs[i][0]);
            f[i + 1] = std::max(f[i], f[j + 1] + jobs[i][2]);
        }

        return f[n];
    }

private:
    // 返回 endtime <= upper 的最大下标
    int search(std::vector<std::vector<int>>& jobs, int right, int upper) {
        int left = -1;
        while (left + 1 < right) {
            int mid = left + (right - left) / 2;
            if (jobs[mid][1] <= upper) {
                left = mid;
            } else {
                right = mid;
            }
        }

        return left;
    }
};
// @lc code=end

