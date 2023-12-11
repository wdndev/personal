// [LCR 180. 文件组合 - 力扣（LeetCode）](https://leetcode.cn/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/description/)

// ```.properties
// 待传输文件被切分成多个部分，按照原排列顺序，每部分文件编号均为一个 正整数（至少含有两个文件）。传输要求为：连续文件编号总和为接收方指定数字 target 的所有文件。请返回所有符合该要求的文件传输组合列表。

// 注意，返回时需遵循以下规则：

// 每种组合按照文件编号 升序 排列；
// 不同组合按照第一个文件编号 升序 排列。
 

// 示例 1：

// 输入：target = 12
// 输出：[[3, 4, 5]]
// 解释：在上述示例中，存在一个连续正整数序列的和为 12，为 [3, 4, 5]。
// ```

class Solution {
public:
    // 1.枚举:枚举每个正整数为起点，判断以它为起点的序列和 sum 是否等于 target即可，
    // 由于题目要求文件组合长度至少大于 2，所以枚举的上界为 (target - 2)/2
    vector<vector<int>> fileCombination1(int target) {
        std::vector<std::vector<int>> ans;
        std::vector<int> tmp_vec;
        int sum = 0;
        for (int i = 1; i <= (target - 1) / 2; i++) {
            for (int j = i; ; j++) {
                sum += j;
                if (sum > target) {
                    sum = 0;
                    break;
                } else if (sum == target) {
                    tmp_vec.clear();
                    for (int k = i; k <= j; k++) {
                        tmp_vec.push_back(k);
                    }
                    ans.push_back(tmp_vec);
                    sum = 0;
                    break;
                }
            }
        }

        return ans;
    }

    // 2.滑动窗口
    vector<vector<int>> fileCombination(int target) {
        std::vector<std::vector<int>> ans;
        std::vector<int> tmp_vec;

        int left = 1;
        int right = 2;
        while ( left < right) {
            int sum = (left + right) * (right - left + 1) / 2;
            // 刚好等于
            if (sum == target) {
                tmp_vec.clear();
                for (int k = left; k <= right; k++) {
                    tmp_vec.push_back(k);
                }
                ans.push_back(tmp_vec);
                left++;
            } else if (sum < target) {
                // 小了，扩大窗口
                right++;
            } else {
                // 大了，扩大窗口
                left++;
            }
        }

        return ans;
    }
};