/*
 * @lc app=leetcode.cn id=239 lang=cpp
 *
 * [239] 滑动窗口最大值
 *
 * https://leetcode.cn/problems/sliding-window-maximum/description/
 *
 * algorithms
 * Hard (49.33%)
 * Likes:    2570
 * Dislikes: 0
 * Total Accepted:    504K
 * Total Submissions: 1M
 * Testcase Example:  '[1,3,-1,-3,5,3,6,7]\n3'
 *
 * 给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k
 * 个数字。滑动窗口每次只向右移动一位。
 * 
 * 返回 滑动窗口中的最大值 。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
 * 输出：[3,3,5,5,6,7]
 * 解释：
 * 滑动窗口的位置                最大值
 * ---------------               -----
 * [1  3  -1] -3  5  3  6  7       3
 * ⁠1 [3  -1  -3] 5  3  6  7       3
 * ⁠1  3 [-1  -3  5] 3  6  7       5
 * ⁠1  3  -1 [-3  5  3] 6  7       5
 * ⁠1  3  -1  -3 [5  3  6] 7       6
 * ⁠1  3  -1  -3  5 [3  6  7]      7
 * 
 * 
 * 示例 2：
 * 
 * 
 * 输入：nums = [1], k = 1
 * 输出：[1]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= nums.length <= 10^5
 * -10^4 <= nums[i] <= 10^4
 * 1 <= k <= nums.length
 * 
 * 
 */

// @lc code=start

// 1.暴力 O(n*k)
// 2.deque O(n)
class Solution {
public:
    // 1.暴力, 超时
    vector<int> maxSlidingWindow1(vector<int>& nums, int k) {
        std::vector<int> ans;
        if (nums.size() != 0) {
            int tmp_max = this->get_vector_max(nums, 0, k);
            for (int i = 0; i < nums.size() - (k - 1); i++) {
                tmp_max = this->get_vector_max(nums, i, i + k);
                ans.push_back(tmp_max);
            }
        }

        return ans;
    }

    // 2.队列
    // 可以使用一个队列存储所有还没有被移除的下标。
    // 在队列中，这些下标按照从小到大的顺序被存储，
    // 并且它们在数组 nums 中对应的值是严格单调递减的。
    // 因为如果队列中有两个相邻的下标，它们对应的值相等或者递增，
    // 那么令前者为 i，后者为 j，就对应了上面所说的情况，
    // 即 nums[i] 会被移除，这就产生了矛盾。

    // 当滑动窗口向右移动时，我们需要把一个新的元素放入队列中。
    // 为了保持队列的性质，我们会不断地将新的元素与队尾的元素相比较，
    // 如果前者大于等于后者，那么队尾的元素就可以被永久地移除，
    // 我们将其弹出队列。我们需要不断地进行此项操作，
    // 直到队列为空或者新的元素小于队尾的元素。

    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int n = nums.size();
        std::deque<int> que;

        // 将前k个元素的下标加入队列中，其中队列头为最大值，队列尾为最小值
        for (int i = 0; i <k; i++) {
            // 将小于队列低的元素加在后面
            while (!que.empty() && nums[i] >= nums[que.back()]) {
                que.pop_back();
            }

            que.push_back(i);
        }

        // 将前k个元素的最大值加进去
        std::vector<int> ans = {nums[que.front()]};
        // 开始遍历
        for (int i = k; i < n; i++) {
            while (!que.empty() && nums[i] >= nums[que.back()]) {
                que.pop_back();
            }

            que.push_back(i);
            //
            while (que.front() <= i - k) {
                que.pop_front();
            }
            ans.push_back(nums[que.front()]);
        }

        return ans;        
    }

private:
    int get_vector_max(std::vector<int>& nums, int start, int end) {
        int max_val = INT_MIN;
        for (int i = start; i <= end; i++) {
            max_val = std::max(max_val, nums[i]);
        }

        return max_val;
    }
};
// @lc code=end

