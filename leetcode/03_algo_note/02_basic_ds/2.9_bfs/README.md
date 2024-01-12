# 9.广度优先搜索

## 1.广度优先搜索

### 1.1 简介

> **广度优先搜索算法（Breadth First Search）**：英文缩写为 BFS，又译作宽度优先搜索 / 横向优先搜索，是一种用于搜索树或图结构的算法。广度优先搜索算法从起始节点开始，逐层扩展，先访问离起始节点最近的节点，后访问离起始节点稍远的节点。以此类推，直到完成整个搜索过程。

因为遍历到的节点顺序符合「先进先出」的特点，所以广度优先搜索可以通过「队列」来实现。

### 1.2 算法步骤

以一个无向图为例，介绍一下广度优先搜索的算法步骤。

1.  将起始节点 `u` 放入队列中，并标记为已访问。
2.  从队列中取出一个节点，访问它并将其所有的未访问邻接节点 `v` 放入队列中。
3.  标记已访问的节点 `v`，以避免重复访问。
4.  重复步骤 2∼3，直到队列为空或找到目标节点。

## 2.队列实现广度优先搜索

1.  定义 $graph$ 为存储无向图的嵌套数组变量，$visited$ 为标记访问节点的集合变量，$queue$ 为存放节点的队列，$u$ 为开始节点，定义 `def bfs(graph, u):` 为队列实现的广度优先搜索方法。
2.  首先将起始节点 $u$ 标记为已访问，并将其加入队列中，即 `visited.add(u)`，`queue.append(u)`。
3.  从队列中取出队头节点 $u$。访问节点 $u$，并对节点进行相关操作（看具体题目要求）。
4.  遍历节点 $u$ 的所有未访问邻接节点 $v$（节点 $v$ 不在 $visited$ 中）。
5.  将节点 $v$ 标记已访问，并加入队列中，即 `visited.add(v)`，`queue.append(v)`。
6.  重复步骤 $3 \sim 5$，直到队列 $queue$ 为空。

```python
import collections

class Solution:
    def bfs(self, graph, u):
        visited = set()                     # 使用 visited 标记访问过的节点
        queue = collections.deque([])       # 使用 queue 存放临时节点
        
        visited.add(u)                      # 将起始节点 u 标记为已访问
        queue.append(u)                     # 将起始节点 u 加入队列中
        
        while queue:                        # 队列不为空
            u = queue.popleft()             # 取出队头节点 u
            print(u)                        # 访问节点 u
            for v in graph[u]:              # 遍历节点 u 的所有未访问邻接节点 v
                if v not in visited:        # 节点 v 未被访问
                    visited.add(v)          # 将节点 v 标记为已访问
                    queue.append(v)         # 将节点 v 加入队列中
                

graph = {
    "0": ["1", "2"],
    "1": ["0", "2", "3"],
    "2": ["0", "1", "3", "4"],
    "3": ["1", "2", "4", "5"],
    "4": ["2", "3"],
    "5": ["3", "6"],
    "6": []
}

# 基于队列实现的广度优先搜索
Solution().bfs(graph, "0")

```

## 3.实战题目

### 3.1 克隆图

[133. 克隆图 - 力扣（LeetCode）](https://leetcode.cn/problems/clone-graph/description/ "133. 克隆图 - 力扣（LeetCode）")

```python
给你无向 连通 图中一个节点的引用，请你返回该图的 深拷贝（克隆）。

图中的每个节点都包含它的值 val（int） 和其邻居的列表（list[Node]）。

class Node {
    public int val;
    public List<Node> neighbors;
}

输入：adjList = [[2,4],[1,3],[2,4],[1,3]]
输出：[[2,4],[1,3],[2,4],[1,3]]
解释：
图中有 4 个节点。
节点 1 的值是 1，它有两个邻居：节点 2 和 4 。
节点 2 的值是 2，它有两个邻居：节点 1 和 3 。
节点 3 的值是 3，它有两个邻居：节点 2 和 4 。
节点 4 的值是 4，它有两个邻居：节点 1 和 3 。

```

广度优先搜索

1.  使用哈希表 $visited$ 来存储原图中被访问过的节点和克隆图中对应节点，键值对为「原图被访问过的节点：克隆图中对应节点」。使用队列 $queue$ 存放节点。
2.  根据起始节点 $node$，创建一个新的节点，并将其添加到哈希表 $visited$ 中，即 `visited[node] = Node(node.val, [])`。然后将起始节点放入队列中，即 `queue.append(node)`。
3.  从队列中取出第一个节点 $node\underline{}u$。访问节点 $node\underline{}u$。
4.  遍历节点 $node\underline{}u$ 的所有未访问邻接节点 $node\underline{}v$（节点 $node\underline{}v$ 不在 $visited$ 中）。
5.  根据节点 $node\underline{}v$ 创建一个新的节点，并将其添加到哈希表 $visited$ 中，即 `visited[node_v] = Node(node_v.val, [])`。
6.  然后将节点 $node\underline{}v$ 放入队列 $queue$ 中，即 `queue.append(node_v)`。
7.  重复步骤 $3 \sim 6$，直到队列 $queue$ 为空。
8.  广度优先搜索结束，返回起始节点的克隆节点（即 $visited[node]$）。

```c++
class Solution {
public:
    // bfs
    Node* cloneGraph(Node* node) {
        if (node == nullptr) {
            return node;
        }
        // 哈希表
        std::unordered_map<Node*, Node*> visited;
        visited[node] = new Node(node->val);
        // 队列
        std::queue<Node*> queue;
        queue.push(node);

        while (!queue.empty()) {
            // 取出结点
            Node* tmp_node = queue.front();
            queue.pop();
            // 遍历邻居
            for (auto& node_v : tmp_node->neighbors) {
                // 如果没有被访问过，克隆，存储在哈希表中
                if (visited.find(node_v) == visited.end()) {
                    visited[node_v] = new Node(node_v->val);
                    queue.push(node_v);

                }
                // 更新当前邻居结点
                visited[tmp_node]->neighbors.push_back(visited[node_v]);
            }

        }

        return visited[node];
    }
private:
    // 访问过结点的标记
    std::unordered_map<Node*, Node*> m_visited;
    // dfs
    Node* dfs(Node* node) {
        // 如果该结点被访问过了，则直接从哈希表中取出克隆的结点
        if (m_visited.find(node) != m_visited.end()) {
            return m_visited[node];
        }

        // 克隆结点，注意，不会克隆邻居
        Node* clone_node = new Node(node->val);
        // 哈希表存储
        m_visited[node] = clone_node;
        // 遍历该节点的邻居节点，并更新克隆结点的邻居结点
        for (auto& neighbor : node->neighbors) {
            clone_node->neighbors.push_back(this->dfs(neighbor));
        }

        return clone_node;
    }
};
```

### 3.2 岛屿的周长

[463. 岛屿的周长 - 力扣（LeetCode）](https://leetcode.cn/problems/island-perimeter/ "463. 岛屿的周长 - 力扣（LeetCode）")

```python
给定一个 row x col 的二维网格地图 grid ，其中：grid[i][j] = 1 表示陆地， grid[i][j] = 0 表示水域。

网格中的格子 水平和垂直 方向相连（对角线方向不相连）。整个网格被水完全包围，但其中恰好有一个岛屿（或者说，一个或多个表示陆地的格子相连组成的岛屿）。

岛屿中没有“湖”（“湖” 指水域在岛屿内部且不和岛屿周围的水相连）。格子是边长为 1 的正方形。网格为长方形，且宽度和高度均不超过 100 。计算这个岛屿的周长。

输入：grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
输出：16
解释：它的周长是上面图片中的 16 个黄色的边

```

1.  使用整形变量 `count` 存储周长，使用队列 `queue` 用于进行广度优先搜索。
2.  遍历一遍二维数组 `grid`，对 `grid[row][col] == 1` 的区域进行广度优先搜索。
3.  先将起始点 `(row, col)` 加入队列。
4.  如果队列不为空，则取出队头坐标 `(row, col)`。先将 `(row, col)` 标记为 `2`，避免重复统计。
5.  然后遍历上、下、左、右四个方向的相邻区域，如果遇到边界或者水域，则周长加 1。
6.  如果相邻区域 `grid[new_row][new_col] == 1`，则将其赋值为 `2`，并将坐标加入队列。
7.  继续执行 4 \~ 6 步，直到队列为空时返回 `count`。

```c++
class Solution {
public:
    // 广度优先搜索
    int islandPerimeter(vector<vector<int>>& grid) {
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[i].size(); j++) {
                if (grid[i][j] == 1) {
                    return this->bfs(grid, i, j);
                }
            }
        }
        return -1;
    }

private:
    // 方向
    int m_dx[4] = {-1, 1, 0, 0};
    int m_dy[4] = {0, 0, -1, 1};

    int bfs(vector<vector<int>>& grid, int i, int j) {
        std::queue<std::pair<int, int>> queue;
        queue.push({i, j});
        int count = 0;
        while (!queue.empty()) {
            auto [row, col] = queue.front();
            queue.pop();
            // 避免重复计算
            grid[row][col] = 2;
            for (int k = 0; k < 4; k++) {
                int x = row + m_dx[k];
                int y = col + m_dy[k];
                // 遇到边界或水域，周长加 1
                if (x < 0 || x >= grid.size() || y < 0 || y >= grid[x].size() || grid[x][y] == 0) {
                    count++;
                // 相邻区域为陆地，则将其标记为2，加入队列
                } else if (grid[x][y] == 1) {
                    grid[x][y] = 2;
                    queue.push({x, y});
                }
            }
        }

        return count;
    }
};
```

### 3.3 打开转盘锁

[752. 打开转盘锁 - 力扣（LeetCode）](https://leetcode.cn/problems/open-the-lock/description/ "752. 打开转盘锁 - 力扣（LeetCode）")

```python
你有一个带有四个圆形拨轮的转盘锁。每个拨轮都有10个数字： '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 。每个拨轮可以自由旋转：例如把 '9' 变为 '0'，'0' 变为 '9' 。每次旋转都只能旋转一个拨轮的一位数字。

锁的初始数字为 '0000' ，一个代表四个拨轮的数字的字符串。

列表 deadends 包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，这个锁将会被永久锁定，无法再被旋转。

字符串 target 代表可以解锁的数字，你需要给出解锁需要的最小旋转次数，如果无论如何不能解锁，返回 -1 。

 

示例 1:

输入：deadends = ["0201","0101","0102","1212","2002"], target = "0202"
输出：6
解释：
可能的移动序列为 "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202"。
注意 "0000" -> "0001" -> "0002" -> "0102" -> "0202" 这样的序列是不能解锁的，
因为当拨动到 "0102" 时这个锁就会被锁定。
```

广度优先搜索

1.  定义 `visited` 为标记访问节点的 set 集合变量，`queue` 为存放节点的队列。
2.  将`0000` 状态标记为访问，并将其加入队列 `queue`。
3.  将当前队列中的所有状态依次出队，判断这些状态是否为死亡字符串。
    1.  如果为死亡字符串，则跳过该状态，否则继续执行。
    2.  如果为目标字符串，则返回当前路径长度，否则继续执行。
4.  枚举当前状态所有位置所能到达的所有状态（通过向上或者向下旋转），并判断是否访问过该状态。
5.  如果之前出现过该状态，则继续执行，否则将其存入队列，并标记访问。
6.  遍历完步骤 3 中当前队列中的所有状态，令路径长度加 `1`，继续执行 3 \~ 5 步，直到队列为空。
7.  如果队列为空，也未能到达目标状态，则返回 `-1`。

```c++
class Solution {
public:
    int openLock(vector<string>& deadends, string target) {
        if (target == "0000") {
            return 0;
        }

        std::unordered_set<std::string> dead(deadends.begin(), deadends.end());
        if (dead.count("0000")) {
            return -1;
        }
        
        auto num_prev = [](char x) -> char {
            return (x == '0' ? '9' : x - 1);
        };
        auto num_succ = [](char x) -> char {
            return (x == '9' ? '0' : x + 1);
        };

        // 枚举status,通过一次旋转得的数字
        auto get_next_status = [&](std::string& status) -> std::vector<std::string> {
            std::vector<std::string> ret;
            for (int i = 0; i < 4; i++) {
                char x = status[i];
                status[i] = num_prev(x);
                ret.push_back(status);
                status[i] = num_succ(x);
                ret.push_back(status);
                status[i] = x;
            }
            return ret;
        };


        std::unordered_set<std::string> visited = {"0000"};

        std::queue<std::pair<std::string, int>> queue;
        queue.push({"0000", 0});

        while (!queue.empty()) {
            auto [status, step] = queue.front();
            queue.pop();
            for (auto& next_status : get_next_status(status)) {
                if (!visited.count(next_status) && !dead.count(next_status)) {
                    if (next_status == target) {
                        return step + 1;
                    }
                    queue.push({next_status, step + 1});
                    visited.insert(std::move(next_status));
                }
            }
        }

        return -1;
    }
};
```

### 3.4 完全平方数

[279. 完全平方数 - 力扣（LeetCode）](https://leetcode.cn/problems/perfect-squares/description/ "279. 完全平方数 - 力扣（LeetCode）")

```python
给你一个整数 n ，返回 和为 n 的完全平方数的最少数量 。

完全平方数 是一个整数，其值等于另一个整数的平方；换句话说，其值等于一个整数自乘的积。例如，1、4、9 和 16 都是完全平方数，而 3 和 11 不是。
 

示例 1：

输入：n = 12
输出：3 
解释：12 = 4 + 4 + 4
```

**1、动态规划**

首先初始化长度为 n+1 的数组 dp，每个位置都为 0

对数组进行遍历，下标为 i，每次都将当前数字先更新为最大的结果，即 `dp[i]=i`，比如 i=4，最坏结果为 4=1+1+1+1 即为 4 个数字

动态转移方程为：`dp[i] = MIN(dp[i], dp[i - j * j] + 1)`，`i `表示当前数字，`j*j` 表示平方数

-   状态定义：`f[i] `表示最少需要多少个数的平方来表示整数 i。
-   转移方程：`f[i] = MIN(f[i], f[i - j * j] + 1)`，其中，$1<j<\sqrt i$，

```c++
class Solution {
public:
    // 1.动态规划
    int numSquares(int n) {
        std::vector<int> dp(n + 1, 0);

        for (int i = 1; i <= n; i++) {
            dp[i] = i;
            for (int j = 1; i - j * j >= 0; j++) {
                dp[i] = std::min(dp[i], dp[i - j*j] + 1);
            }
        }
        return dp[n];
    }
};
```

### 3.5 01矩阵

[542. 01 矩阵 - 力扣（LeetCode）](https://leetcode.cn/problems/01-matrix/description/ "542. 01 矩阵 - 力扣（LeetCode）")

```python
给定一个由 0 和 1 组成的矩阵 mat ，请输出一个大小相同的矩阵，其中每一个格子是 mat 中对应位置元素到最近的 0 的距离。

两个相邻元素间的距离为 1 。

输入：mat = [[0,0,0],[0,1,0],[0,0,0]]
输出：[[0,0,0],[0,1,0],[0,0,0]]

```

广度优先搜索

题目要求的是每个 `1` 到 `0`的最短曼哈顿距离。

将所有值为 `0` 的元素位置保存到队列中，然后对所有值为 `0` 的元素开始进行广度优先搜索，每搜一步距离加 `1`，当每次搜索到 `1` 时，就可以得到 `0` 到这个 `1` 的最短距离，也就是当前离这个 `1` 最近的 `0` 的距离。

具体步骤如下：

1.  使用一个集合变量 `visited` 存储所有值为 `0` 的元素坐标。使用队列变量 `queue` 存储所有值为 `0` 的元素坐标。使用二维数组 `res` 存储对应位置元素（即 ���\[�]\[�]）到最近的 0 的距离。
2.  我们从所有为如果队列 `queue` 不为空，则从队列中依次取出值为 `0` 的元素坐标，遍历其上、下、左、右位置。
3.  如果相邻区域未被访问过（说明遇到了值为 `1` 的元素），则更新相邻位置的距离值，并把相邻位置坐标加入队列 `queue` 和访问集合 `visited` 中。
4.  继续执行 2 \~ 3 步，直到队列为空时，返回 `res`。

```c++
class Solution {
public:
    vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
        int rows = mat.size();
        int cols = mat[0].size();

        std::vector<std::vector<int>> ans(rows, std::vector<int>(cols, 0));
        // std::unordered_set<std::pair<int, int> > visited;
        std::vector<std::vector<int>> visited(rows, std::vector<int>(cols, 0));

         std::queue<std::pair<int, int>> queue;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (mat[i][j] == 0) {
                    visited[i][j] = 1;
                    queue.push({i, j});
                }
            }
        }

        int dx[4] = {-1, 1, 0, 0};
        int dy[4] = {0, 0, -1, 1};


        while (!queue.empty()) {
            auto [i, j] = queue.front();
            queue.pop();

            for (int k = 0; k < 4; k++) {
                int x = i + dx[k];
                int y = j + dy[k];

                if (x >= 0 && x < rows && y >= 0 && y < cols && !visited[x][y]) {
                    ans[x][y] = ans[i][j] + 1;
                    queue.push({x, y});
                    visited[x][y] = 1;
                }
            }
        }

        return ans;
    }
};
```

### 3.6 零钱兑换

[322. 零钱兑换 - 力扣（LeetCode）](https://leetcode.cn/problems/coin-change/description/ "322. 零钱兑换 - 力扣（LeetCode）")

```python
给你一个整数数组 coins ，表示不同面额的硬币；以及一个整数 amount ，表示总金额。

计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额，返回 -1 。

你可以认为每种硬币的数量是无限的。

 

示例 1：

输入：coins = [1, 2, 5], amount = 11
输出：3 
解释：11 = 5 + 5 + 1
```

DP方法

1.  分治（子问题）：$f[n] = min ~\{f(n-k), for ~k ~in [1, 2, 5]\} + 1$
2.  状态数组定义：$f(n)$，凑成总金额为 n 的最少硬币数量。
3.  DP方程：$f[n] = min~ \{f(n-k), for ~k ~in [1, 2, 5]\} + 1$

```c++
class Solution {
public:
    // 1.动态规划
    int coinChange(vector<int>& coins, int amount) {
        int max_amount = amount + 1;
        std::vector<int> dp(amount + 1, max_amount);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++) {
            for (int j = 0; j < coins.size(); j++) {
                if (coins[j] <= i) {
                    dp[i] = std::min(dp[i], dp[i - coins[j]] + 1);
                }
            }
        }

        return dp[amount] > amount ? -1 : dp[amount];
    }
};
```

### 3.7 衣橱整理

[LCR 130. 衣橱整理 - 力扣（LeetCode）](https://leetcode.cn/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/description/ "LCR 130. 衣橱整理 - 力扣（LeetCode）")

```python
家居整理师将待整理衣橱划分为 m x n 的二维矩阵 grid，其中 grid[i][j] 代表一个需要整理的格子。整理师自 grid[0][0] 开始 逐行逐列 地整理每个格子。

整理规则为：在整理过程中，可以选择 向右移动一格 或 向下移动一格，但不能移动到衣柜之外。同时，不需要整理 digit(i) + digit(j) > cnt 的格子，其中 digit(x) 表示数字 x 的各数位之和。

请返回整理师 总共需要整理多少个格子。

 

示例 1：

输入：m = 4, n = 7, cnt = 5
输出：18
```

先定义一个计算数位和的方法 `digitsum`，该方法输入一个整数，返回该整数各个数位的总和。

然后使用广度优先搜索方法，具体步骤如下：

-   将 `(0, 0)` 加入队列 `queue` 中。
-   当队列不为空时，每次将队首坐标弹出，加入访问集合 `visited` 中。
-   再将满足行列坐标的数位和不大于 `k` 的格子位置加入到队列中，继续弹出队首位置。
-   直到队列为空时停止。输出访问集合的长度。

```c++
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
```
