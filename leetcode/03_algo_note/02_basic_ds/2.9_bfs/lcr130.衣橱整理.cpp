// ### 3.7 衣橱整理

// [LCR 130. 衣橱整理 - 力扣（LeetCode）](https://leetcode.cn/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/description/)

// ```Python
// 家居整理师将待整理衣橱划分为 m x n 的二维矩阵 grid，其中 grid[i][j] 代表一个需要整理的格子。整理师自 grid[0][0] 开始 逐行逐列 地整理每个格子。

// 整理规则为：在整理过程中，可以选择 向右移动一格 或 向下移动一格，但不能移动到衣柜之外。同时，不需要整理 digit(i) + digit(j) > cnt 的格子，其中 digit(x) 表示数字 x 的各数位之和。

// 请返回整理师 总共需要整理多少个格子。

 

// 示例 1：

// 输入：m = 4, n = 7, cnt = 5
// 输出：18
// ```

// 先定义一个计算数位和的方法 `digitsum`，该方法输入一个整数，返回该整数各个数位的总和。

// 然后使用广度优先搜索方法，具体步骤如下：

// - 将 `(0, 0)` 加入队列 `queue` 中。
// - 当队列不为空时，每次将队首坐标弹出，加入访问集合 `visited` 中。
// - 再将满足行列坐标的数位和不大于 `k` 的格子位置加入到队列中，继续弹出队首位置。
// - 直到队列为空时停止。输出访问集合的长度。


class Solution {
public:
    int wardrobeFinishing(int m, int n, int cnt) {
        if (!cnt) return 1;
        std::queue<std::pair<int, int>> queue;
        std::vector<std::vector<int>> visited(m, std::vector<int>(n, 0));
        queue.push({0, 0});
        visited[0][0] = 1;
        int count = 1;
        while (!queue.empty()) {
            auto [i, j] = queue.front();
            queue.pop();
            for (int k = 0; k < 2; k++) {
                int x = i + m_dx[k];
                int y = j + m_dy[k];
                if (x >= 0 && x < m && y >= 0 && y < n 
                    && this->digitsum(x) + this->digitsum(y) <= cnt 
                    && !visited[x][y]) {
                    queue.push({x, y});
                    visited[x][y] = 1;
                    count++;
                }
            }

        }
        return count;
    }
private:
    // 方向
    int m_dx[2] = {1, 0};
    int m_dy[2] = {0, 1};
    int digitsum1(int num) {
        int sum = 0;
        while (num) {
            sum += num % 10;
            num = num / 10;
        }
        return sum;
    }
    int digitsum(int x) {
        int res=0;
        for (; x; x /= 10) {
            res += x % 10;
        }
        return res;
    }

};
