# 6.深度优先搜索

## 1.深度优先搜索简介

> **深度优先搜索算法（Depth First Search）**：英文缩写为 DFS，**是一种用于搜索树或图结构的算法**。深度优先搜索算法采用了回溯思想，从起始节点开始，沿着一条路径尽可能深入地访问节点，直到无法继续前进时为止，然后回溯到上一个未访问的节点，继续深入搜索，直到完成整个搜索过程。

深度优先搜索算法中所谓的深度优先，就是说优先沿着一条路径走到底，直到无法继续深入时再回头。

在深度优先遍历的过程中，需要将当前遍历节点 $u$ 的相邻节点暂时存储起来，以便于在回退的时候可以继续访问它们。遍历到的节点顺序符合「后进先出」的特点，这正是「递归」和「堆栈」所遵循的规律，所以深度优先搜索可以通过「递归」或者「堆栈」来实现。

## 2.算法步骤及实现

### 2.1 算法步骤

以一个无向图为例，介绍一下深度优先搜索的算法步骤。

1.  选择起始节点 $u$，并将其标记为已访问。
2.  检查当前节点是否为目标节点（看具体题目要求）。
3.  如果当前节点 $u$ 是目标节点，则直接返回结果。
4.  如果当前节点 $u$ 不是目标节点，则遍历当前节点 $u$ 的所有未访问邻接节点。
5.  对每个未访问的邻接节点 $v$，从节点 $v$ 出发继续进行深度优先搜索（递归）。
6.  如果节点 $u$ 没有未访问的相邻节点，回溯到上一个节点，继续搜索其他路径。
7.  重复 $2 \sim 6$ 步骤，直到遍历完整个图或找到目标节点为止。

### 2.2 基于递归实现DFS

1.  定义 $graph$ 为存储无向图的嵌套数组变量，$visited$ 为标记访问节点的集合变量。$u$ 为当前遍历边的开始节点。定义 `def dfs_recursive(graph, u, visited):` 为递归实现的深度优先搜索方法。
2.  选择起始节点 $u$，并将其标记为已访问，即将节点 $u$ 放入 $visited$ 中（`visited.add(u)`）。
3.  检查当前节点 $u$ 是否为目标节点（看具体题目要求）。
4.  如果当前节点 $u$ 是目标节点，则直接返回结果。
5.  如果当前节点 $u$ 不是目标节点，则遍历当前节点 $u$ 的所有未访问邻接节点。
6.  对每个未访问的邻接节点 $v$，从节点 $v$ 出发继续进行深度优先搜索（递归），即调用 `dfs_recursive(graph, v, visited)`。
7.  如果节点 $u$ 没有未访问的相邻节点，则回溯到最近访问的节点，继续搜索其他路径。
8.  重复 $3 \sim 7$ 步骤，直到遍历完整个图或找到目标节点为止。

```python
class Solution:
    def dfs_recursive(self, graph, u, visited):
        print(u)                        # 访问节点
        visited.add(u)                  # 节点 u 标记其已访问

        for v in graph[u]:
            if v not in visited:        # 节点 v 未访问过
                # 深度优先搜索遍历节点
                self.dfs_recursive(graph, v, visited)
        

graph = {
    "A": ["B", "C"],
    "B": ["A", "C", "D"],
    "C": ["A", "B", "D", "E"],
    "D": ["B", "C", "E", "F"],
    "E": ["C", "D"],
    "F": ["D", "G"],
    "G": []
}

# 基于递归实现的深度优先搜索
visited = set()
Solution().dfs_recursive(graph, "A", visited)
```

### 2.3 基于堆栈实现DFS

为了防止多次遍历同一节点，在使用栈存放节点访问记录时，将「当前节点」以及「下一个将要访问的邻接节点下标」一同存入栈中，从而在出栈时，可以通过下标直接找到下一个邻接节点，而不用遍历所有邻接节点。

1.  定义 $graph$ 为存储无向图的嵌套数组变量，$visited$ 为标记访问节点的集合变量。$start$ 为当前遍历边的开始节点。定义 $stack$ 用于存放节点访问记录的栈结构。
2.  选择起始节点 $u$，检查当前节点 $u$ 是否为目标节点（看具体题目要求）。
3.  如果当前节点 $u$ 是目标节点，则直接返回结果。
4.  如果当前节点 $u$ 不是目标节点，则将节点 $u$ 以及节点 $u$ 下一个将要访问的邻接节点下标 $0$ 放入栈中，并标记为已访问，即 `stack.append([u, 0])`，`visited.add(u)`。
5.  如果栈不为空，取出 $stack$ 栈顶元素节点 $u$，以及节点 $u$ 下一个将要访问的邻接节点下标 $i$。
6.  根据节点 $u$ 和下标 $i$，取出将要遍历的未访问过的邻接节点 $v$。
7.  将节点 $u$ 以及节点 u 的下一个邻接节点下标 $i + 1$ 放入栈中。
8.  访问节点 $v$，并对节点进行相关操作（看具体题目要求）。
9.  将节点 $v$ 以及节点 $v$ 下一个邻接节点下标 $0$ 放入栈中，并标记为已访问，即 `stack.append([v, 0])`，`visited.add(v)`。
10. 重复步骤 $5 \sim 9$，直到 $stack$ 栈为空或找到目标节点为止。

```python
class Solution:
    def dfs_stack(self, graph, u):
        print(u)                            # 访问节点 u
        visited, stack = set(), []          # 使用 visited 标记访问过的节点, 使用栈 stack 存放临时节点
        
        stack.append([u, 0])                # 将节点 u，节点 u 的下一个邻接节点下标放入栈中，下次将遍历 graph[u][0]
        visited.add(u)                      # 将起始节点 u 标记为已访问
        
    
        while stack:
            u, i = stack.pop()              # 取出节点 u，以及节点 u 下一个将要访问的邻接节点下标 i
            
            if i < len(graph[u]):
                v = graph[u][i]             # 取出邻接节点 v
                stack.append([u, i + 1])    # 下一次将遍历 graph[u][i + 1]
                if v not in visited:        # 节点 v 未访问过
                    print(v)                # 访问节点 v
                    stack.append([v, 0])    # 下一次将遍历 graph[v][0]
                    visited.add(v)          # 将节点 v 标记为已访问                
        

graph = {
    "A": ["B", "C"],
    "B": ["A", "C", "D"],
    "C": ["A", "B", "D", "E"],
    "D": ["B", "C", "E", "F"],
    "E": ["C", "D"],
    "F": ["D", "G"],
    "G": []
}

# 基于堆栈实现的深度优先搜索
Solution().dfs_stack(graph, "A")
```

## 3.实战题目

### 3.1 岛屿数量

[200. 岛屿数量 - 力扣（LeetCode）](https://leetcode.cn/problems/number-of-islands/description/ "200. 岛屿数量 - 力扣（LeetCode）")

```python
给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

 

示例 1：

输入：grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
输出：1
```

填海造陆，遍历地图，遇到一个1，将1进行BFS/DFS遍历，将周围的1全部变为0，操作次数记为一次

dfs的数量就是陆地的数量

```c++
class Solution {
public:
    // 填海造陆
    // 遍历地图，遇到一个1，将1进行BFS/DFS遍历，将周围的1全部变为0，操作次数记为一次
    // 操作次数就是岛屿数量
    int numIslands(vector<vector<char>>& grid) {
        int island_num = 0;
        m_grid = grid;

        for (int i = 0; i < m_grid.size(); i++) {
            for (int j = 0; j < m_grid[i].size(); j++) {
                if (m_grid[i][j] == '0') {
                    continue;
                }
                island_num += this->sink(i, j);
            }
        }

        return island_num;
    }
private:
    // 方向
    int m_dx[4] = {-1, 1, 0, 0};
    int m_dy[4] = {0, 0, -1, 1};
    // 全局地图
    std::vector<std::vector<char>> m_grid;

    // bfs遍历
    int sink(int i, int j) {
        if (m_grid[i][j] == '0') {
            return 0;
        }

        // 将i，j变为0
        m_grid[i][j] = '0';
        // 开始遍历四个方向
        for (int k = 0; k < 4; k++) {
            int x = i + m_dx[k];
            int y = j + m_dy[k];

            if (x >=0 && x < m_grid.size() && y >= 0 && y < m_grid[i].size()) {
                // 遇到0跳过
                // 遇到1开始递归
                if (m_grid[x][y] == '0') {
                    continue;
                }
                this->sink(x, y);
            }
        }
        return 1;
    }
};
```

### 3.2 克隆图

[133. 克隆图 - 力扣（LeetCode）](https://leetcode.cn/problems/clone-graph/ "133. 克隆图 - 力扣（LeetCode）")

```python
给你无向 连通 图中一个节点的引用，请你返回该图的 深拷贝（克隆）。

图中的每个节点都包含它的值 val（int） 和其邻居的列表（list[Node]）。

class Node {
    public int val;
    public List<Node> neighbors;
}
 

测试用例格式：

简单起见，每个节点的值都和它的索引相同。例如，第一个节点值为 1（val = 1），第二个节点值为 2（val = 2），以此类推。该图在测试用例中使用邻接列表表示。

邻接列表 是用于表示有限图的无序列表的集合。每个列表都描述了图中节点的邻居集。

给定节点将始终是图中的第一个节点（值为 1）。你必须将 给定节点的拷贝 作为对克隆图的引用返回。

输入：adjList = [[2,4],[1,3],[2,4],[1,3]]
输出：[[2,4],[1,3],[2,4],[1,3]]
解释：
图中有 4 个节点。
节点 1 的值是 1，它有两个邻居：节点 2 和 4 。
节点 2 的值是 2，它有两个邻居：节点 1 和 3 。
节点 3 的值是 3，它有两个邻居：节点 2 和 4 。
节点 4 的值是 4，它有两个邻居：节点 1 和 3 。
```

1.  使用哈希表 `m_visited `来存储原图中被访问过的节点和克隆图中对应节点，键值对为「原图被访问过的节点：克隆图中对应节点」。
2.  从给定节点开始，以深度优先搜索的方式遍历原图。
    1.  如果当前节点被访问过，则返回隆图中对应节点。
    2.  如果当前节点没有被访问过，则创建一个新的节点，并保存在哈希表中。
    3.  遍历当前节点的邻接节点列表，递归调用当前节点的邻接节点，并将其放入克隆图中对应节点。
3.  递归结束，返回克隆节点。

```c++
class Solution {
public:
    Node* cloneGraph(Node* node) {
        if (node == nullptr) {
            return node;
        }

        return this->dfs(node);
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

### 3.3 目标和

[494. 目标和 - 力扣（LeetCode）](https://leetcode.cn/problems/target-sum/ "494. 目标和 - 力扣（LeetCode）")

```python
给你一个非负整数数组 nums 和一个整数 target 。

向数组中的每个整数前添加 '+' 或 '-' ，然后串联起所有整数，可以构造一个 表达式 ：

- 例如，nums = [2, 1] ，可以在 2 之前添加 '+' ，在 1 之前添加 '-' ，然后串联起来得到表达式 "+2-1" 。
返回可以通过上述方法构造的、运算结果等于 target 的不同 表达式 的数目。

 

示例 1：

输入：nums = [1,1,1,1,1], target = 3
输出：5
解释：一共有 5 种方法让最终目标和为 3 。
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3
```

1、深度优先搜索&#x20;

1.  定义从位置 `0`、和为 `0` 开始，到达数组尾部位置为止，和为 `target` 的方案数为 `dfs(0, 0)`。
2.  下面从位置 `0`、和为 `0` 开始，以深度优先搜索遍历每个位置。
3.  如果当前位置 `i` 遍历完所有位置：
    1.  如果和 `cur_sum` 等于目标和 `target`，则返回方案数 `1`。
    2.  如果和 `cur_sum` 不等于目标和 `target`，则返回方案数 `0`。
4.  如果当前位置 `i`、和为 `cur_sum` 之前没有记录过，则：
    1.  递归搜索 `i + 1` 位置，和为 `cur_sum - nums[i]` 的方案数。
    2.  递归搜索 `i + 1` 位置，和为 `cur_sum + nums[i]` 的方案数。
5.  最终方案数为 `dfs(0, 0)`，将其作为答案返回即可。

2、动态规划

假设数组中所有元素和为 `sum`，数组中所有符号为 `+` 的元素为 `sum_x`，符号为 `-` 的元素和为 `sum_y`。则 `target = sum_x - sum_y`。

而 `sum_x + sum_y = sum`。根据两个式子可以求出 `2 * sum_x = target + sum` ，即 `sum_x = (target + sum) / 2`。

那么这道题就变成了，如何在数组中找到一个集合，使集合中元素和为 `(target + sum) / 2`。这就变为了求容量为 `(target + sum) / 2` 的 01 背包问题。

-   状态定义：`dp[i]`: 填满容量为i的背包，有`dp[i]`种方法
-   状态转移方程：`dp[i] = dp[i] + dp[i - num]`。不使用当前 `num`：只使用之前元素填满容量为 `i` 的背包的方法数。使用当前 `num`：填满容量 `i - num` 的包的方法数，再填入 `num` 的方法数。
-   初始化：默认填满容量为 `0` 的背包有 `1` 种办法。即 `dp[i] = 1`

```c++
class Solution {
public:
    // 1.深度优先搜索 + 记忆化搜索
    int findTargetSumWays1(vector<int>& nums, int target) {

        return this->dfs(0, 0, nums, target);
    }
    int dfs(int idx, int curr_sum, std::vector<int>& nums, int target) {
        if (idx == nums.size()) {
            if (curr_sum == target) {
                return 1;
            } else {
                return 0;
            }
        }

        int ans = this->dfs(idx+1, curr_sum-nums[idx], nums, target) 
                + this->dfs(idx+1, curr_sum+nums[idx], nums, target);

        return ans;
    }

    // 2.动态规划
    int findTargetSumWays(vector<int>& nums, int target) {
        int sum_nums = accumulate(nums.begin(), nums.end(), 0);
        if (abs(target) > abs(sum_nums) || (target + sum_nums) % 2 == 1) {
            return 0;
        }
        int size = (target + sum_nums) / 2;
        std::vector<int> dp(size + 1, 0);
        dp[0] = 1;
        for (auto& num : nums) {
            for (int i = size; i  > num - 1; i--) {
                dp[i] = dp[i] + dp[i - num];
            }
        }

        return dp[size];
    }
};
```

### 3.4 钥匙和房间

[841. 钥匙和房间 - 力扣（LeetCode）](https://leetcode.cn/problems/keys-and-rooms/description/ "841. 钥匙和房间 - 力扣（LeetCode）")

```python
有 n 个房间，房间按从 0 到 n - 1 编号。最初，除 0 号房间外的其余所有房间都被锁住。你的目标是进入所有的房间。然而，你不能在没有获得钥匙的时候进入锁住的房间。

当你进入一个房间，你可能会在里面找到一套不同的钥匙，每把钥匙上都有对应的房间号，即表示钥匙可以打开的房间。你可以拿上所有钥匙去解锁其他房间。

给你一个数组 rooms 其中 rooms[i] 是你进入 i 号房间可以获得的钥匙集合。如果能进入 所有 房间返回 true，否则返回 false。

 

示例 1：

输入：rooms = [[1],[2],[3],[]]
输出：true
解释：
我们从 0 号房间开始，拿到钥匙 1。
之后我们去 1 号房间，拿到钥匙 2。
然后我们去 2 号房间，拿到钥匙 3。
最后我们去了 3 号房间。
由于我们能够进入每个房间，我们返回 true。
```

那么问题就变为了给定一张有向图，从 `0` 节点开始出发，问是否能到达所有的节点。

1.  使用  `visited` 来统计遍历到的节点个数。
2.  从 `0` 节点开始，使用深度优先搜索的方式遍历整个图。
3.  将当前节点 `x` 加入到集合 `visited` 中，遍历当前节点的邻接点。
    1.  如果邻接点不再集合 `visited` 中，则继续递归遍历。
4.  最后深度优先搜索完毕，判断一下遍历到的节点个数是否等于图的节点个数（即集合 `visited` 中的元素个数是否等于节点个数）。
    1.  如果等于，则返回 `True`
    2.  如果不等于，则返回 `False`。

```c++
class Solution {
public:
    bool canVisitAllRooms(vector<vector<int>>& rooms) {
        m_num = 0;
        m_visited.resize(rooms.size());
        this->dfs(0, rooms);
        return m_num == rooms.size();
    }

private:
    std::vector<int> m_visited;
    int m_num;

    void dfs(int x, vector<vector<int>>& rooms) {
        m_visited[x] = true;
        m_num++;
        for (auto& it : rooms[x]) {
            if (!m_visited[it]) {
                this->dfs(it, rooms);
            }
        }

    }
};
```

### 3.5 岛屿的最大面积

[695. 岛屿的最大面积 - 力扣（LeetCode）](https://leetcode.cn/problems/max-area-of-island/description/ "695. 岛屿的最大面积 - 力扣（LeetCode）")

```python
给你一个大小为 m x n 的二进制矩阵 grid 。

岛屿 是由一些相邻的 1 (代表土地) 构成的组合，这里的「相邻」要求两个 1 必须在 水平或者竖直的四个方向上 相邻。你可以假设 grid 的四个边缘都被 0（代表水）包围着。

岛屿的面积是岛上值为 1 的单元格的数目。

计算并返回 grid 中最大的岛屿面积。如果没有岛屿，则返回面积为 0 。

输入：grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
输出：6
解释：答案不应该是 11 ，因为岛屿只能包含水平或垂直这四个方向上的 1 。

```

填海造陆，遍历地图，遇到一个1，将1进行BFS/DFS遍历，将周围的1全部变为0，记录最大岛屿面积

取最大岛屿面积

```c++
class Solution {
public:
    // 填海造陆
    // 遍历地图，遇到一个1，将1进行BFS/DFS遍历，将周围的1全部变为0，记录最大岛屿面积
    // 取最大岛屿面积
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        int max_area = 0;
        m_grid = grid;

        for (int i = 0; i < m_grid.size(); i++) {
            for (int j = 0; j < m_grid[i].size(); j++) {
                if (m_grid[i][j] == 0) {
                    continue;
                }
                max_area = std::max(max_area, this->sink(i, j));
            }
        }

        return max_area;
    }
private:
    // 方向
    int m_dx[4] = {-1, 1, 0, 0};
    int m_dy[4] = {0, 0, -1, 1};
    // 全局地图
    std::vector<std::vector<int>> m_grid;

    // bfs遍历
    int sink(int i, int j) {
        if (m_grid[i][j] == 0) {
            return 0;
        }
        int area = 1;
        // 将i，j变为0
        m_grid[i][j] = 0;
        // 开始遍历四个方向
        for (int k = 0; k < 4; k++) {
            int x = i + m_dx[k];
            int y = j + m_dy[k];

            if (x >=0 && x < m_grid.size() && y >= 0 && y < m_grid[i].size()) {
                // 遇到0跳过
                // 遇到1开始递归
                if (m_grid[x][y] == 0) {
                    continue;
                }
                area += this->sink(x, y);
            }
        }
        return area;
    }
};
```

### 3.6 被围绕的区域

[130. 被围绕的区域 - 力扣（LeetCode）](https://leetcode.cn/problems/surrounded-regions/description/ "130. 被围绕的区域 - 力扣（LeetCode）")

```python
给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' ，找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。


输入：board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
输出：[["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
解释：被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上，或不与边界上的 'O' 相连的 'O' 最终都会被填充为 'X'。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。

```

深度优先搜索

根据题意，任何边界上的 `O` 都不会被填充为`X`。而被填充 `X` 的 `O` 一定在内部不在边界上。

所以可以用深度优先搜索先搜索边界上的 `O` 以及与边界相连的 `O`，将其先标记为 `#`。

最后遍历一遍 `board`，将所有 `#` 变换为 `O`，将所有 `O` 变换为 `X`。

```c++
class Solution {
public:
    void solve(vector<vector<char>>& board) {
        if (board.empty()) {
            return;
        }
        int rows = board.size();
        int cols = board[0].size();

        for (int i = 0; i < rows; i++) {
            this->dfs(i, 0, board);
            this->dfs(i, cols - 1, board);
        }

        for (int j = 0; j < cols - 1; j++) {
            this->dfs(0, j, board);
            this->dfs(rows - 1, j, board);
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (board[i][j] == '#') {
                    board[i][j] = 'O';
                } else if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
            }
        }

    }

private:
    // 方向
    int m_dx[4] = {-1, 1, 0, 0};
    int m_dy[4] = {0, 0, -1, 1};
    void dfs(int i, int j, vector<vector<char>>& board) {
        if (i < 0 || i >= board.size()
            || j < 0 || j >= board[i].size()
            || board[i][j] != 'O') {
            return;
        }
        board[i][j] = '#';
        for (int k = 0; k < 4; k++) {
            int x = i + m_dx[k];
            int y = j + m_dy[k];
            this->dfs(x, y, board);
        }
    }
};
```

### 3.7 太平洋大西洋流水问题

[417. 太平洋大西洋水流问题 - 力扣（LeetCode）](https://leetcode.cn/problems/pacific-atlantic-water-flow/description/ "417. 太平洋大西洋水流问题 - 力扣（LeetCode）")

```python
有一个 m × n 的矩形岛屿，与 太平洋 和 大西洋 相邻。 “太平洋” 处于大陆的左边界和上边界，而 “大西洋” 处于大陆的右边界和下边界。

这个岛被分割成一个由若干方形单元格组成的网格。给定一个 m x n 的整数矩阵 heights ， heights[r][c] 表示坐标 (r, c) 上单元格 高于海平面的高度 。

岛上雨水较多，如果相邻单元格的高度 小于或等于 当前单元格的高度，雨水可以直接向北、南、东、西流向相邻单元格。水可以从海洋附近的任何单元格流入海洋。

返回网格坐标 result 的 2D 列表 ，其中 result[i] = [ri, ci] 表示雨水从单元格 (ri, ci) 流动 既可流向太平洋也可流向大西洋 。

输入: heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
输出: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]


输入: heights = [[2,1],[1,2]]
输出: [[0,0],[0,1],[1,0],[1,1]]

```

雨水由高处流向低处，如果根据雨水的流向搜索，来判断是否能从某一位置流向太平洋和大西洋不太容易。可以换个思路。

1.  分别从太平洋和大西洋（就是矩形边缘）出发，逆流而上，找出水流逆流能达到的地方，可以用两个二维数组 `pacific`、`atlantic` 分别记录太平洋和大西洋能到达的位置。
2.  使用深度优先搜索实现反向搜索，搜索过程中需要记录每个单元格是否可以从太平洋反向到达以及是否可以从大西洋反向到达
3.  然后再对二维数组进行一次遍历，找出两者交集的位置，就是雨水既可流向太平洋也可流向大西洋的位置，将其加入答案数组 `res` 中。
4.  最后返回答案数组 `res`。

```c++
class Solution {
public:
    vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
        this->m_heights = heights;
        int rows = heights.size();
        int cols = heights[0].size();
        // 太平洋能到达的位置
        std::vector<std::vector<bool>> pacific(rows, std::vector<bool>(cols, false));
        // 大西洋能到达的位置
        std::vector<std::vector<bool>> altlantic(rows, std::vector<bool>(cols, false));

        for (int i = 0; i < rows; i++) {
            this->dfs(i, 0, pacific);
            this->dfs(i, cols - 1, altlantic);
        }
        for (int j = 0; j < cols; j++) {
            this->dfs(0, j, pacific);
            this->dfs(rows - 1, j, altlantic);
        }
        
        std::vector<std::vector<int>> ans;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (pacific[i][j] && altlantic[i][j]) {
                    ans.push_back({i, j});
                }
            }
        }

        return ans;
    }
private:
    // 方向
    int m_dx[4] = {-1, 1, 0, 0};
    int m_dy[4] = {0, 0, -1, 1};
    std::vector<std::vector<int>> m_heights;
    void dfs(int i, int j, std::vector<std::vector<bool>>& visited) {
        if (visited[i][j]) {
            return;
        }
        visited[i][j] = true;
        for (int k = 0; k < 4; k++) {
            int x = i + m_dx[k];
            int y = j + m_dy[k];

            if (x >=0 && x < visited.size() && y >= 0 && y < visited[i].size()
                &&  m_heights[x][y] >= m_heights[i][j]) {
                this->dfs(x, y, visited);
            }
        }
    }
};
```

### 3.8 飞地的数量

[1020. 飞地的数量 - 力扣（LeetCode）](https://leetcode.cn/problems/number-of-enclaves/description/ "1020. 飞地的数量 - 力扣（LeetCode）")

```python
给你一个大小为 m x n 的二进制矩阵 grid ，其中 0 表示一个海洋单元格、1 表示一个陆地单元格。

一次 移动 是指从一个陆地单元格走到另一个相邻（上、下、左、右）的陆地单元格或跨过 grid 的边界。

返回网格中 无法 在任意次数的移动中离开网格边界的陆地单元格的数量。

输入：grid = [[0,0,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,0]]
输出：3
解释：有三个 1 被 0 包围。一个 1 没有被包围，因为它在边界上。

```

深度优先搜索

与四条边界相连的陆地单元是肯定能离开网络边界的。

可以先通过深度优先搜索将与四条边界相关的陆地全部变为海（赋值为 `0`）。

然后统计网格中 `1` 的数量，即为答案。

```c++
class Solution {
public:
    int numEnclaves(vector<vector<int>>& grid) {
        int rows = grid.size();
        int cols = grid[0].size();

        for (int i = 0; i < rows; i++) {
            if (grid[i][0] == 1) {
                this->dfs(i, 0, grid);
            }
            if (grid[i][cols - 1] == 1) {
                this->dfs(i, cols - 1, grid);
            }
        }

        for (int j = 0; j < cols; j++) {
            if (grid[0][j] == 1) {
                this->dfs(0, j, grid);
            }
            if (grid[rows - 1][j] == 1) {
                this->dfs(rows - 1, j, grid);
            }
        }

        int ans = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 1) {
                    ans++;
                }
            }
        }

        return ans;
    }
private:
    // 方向
    int m_dx[4] = {-1, 1, 0, 0};
    int m_dy[4] = {0, 0, -1, 1};

    void dfs(int i, int j, vector<vector<int>>& grid) {
        if (i < 0 || i >= grid.size()
            || j < 0 || j >= grid[i].size()
            || grid[i][j] == 0) {
            return;
        }

        grid[i][j] = 0;

        for (int k = 0; k < 4; k++) {
            int x = i + m_dx[k];
            int y = j + m_dy[k];

            this->dfs(x, y, grid);
        }
    }
};
```

### 3.9 统计封闭岛屿的数目

[1254. 统计封闭岛屿的数目 - 力扣（LeetCode）](https://leetcode.cn/problems/number-of-closed-islands/description/ "1254. 统计封闭岛屿的数目 - 力扣（LeetCode）")

```python
二维矩阵 grid 由 0 （土地）和 1 （水）组成。岛是由最大的4个方向连通的 0 组成的群，封闭岛是一个 完全 由1包围（左、上、右、下）的岛。

请返回 封闭岛屿 的数目。

输入：grid = [[1,1,1,1,1,1,1,0],[1,0,0,0,0,1,1,0],[1,0,1,0,1,1,1,0],[1,0,0,0,0,1,0,1],[1,1,1,1,1,1,1,0]]
输出：2
解释：
灰色区域的岛屿是封闭岛屿，因为这座岛屿完全被水域包围（即被 1 区域包围）。

```

1.  从 `grid[i][j] == 0` 的位置出发，使用深度优先搜索的方法遍历上下左右四个方向上相邻区域情况。
    1.  如果上下左右都是 `grid[i][j] == 1`，则返回 `True`。
    2.  如果有一个以上方向的 `grid[i][j] == 0`，则返回 `False`。
    3.  遍历之后将当前陆地位置置为 `1`，表示该位置已经遍历过了。
2.  最后统计出上下左右都满足 `grid[i][j] == 1` 的情况数量，即为答案。

```c++
class Solution {
public:
    int closedIsland(vector<vector<int>>& grid) {
        int ans = 0;
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                if (grid[i][j] == 0 && this->dfs(i, j, grid)) {
                    ans++;
                }
            }
        }
        return ans;
    }
private:
    // 方向
    int m_dx[4] = {-1, 1, 0, 0};
    int m_dy[4] = {0, 0, -1, 1};

    bool  dfs(int i, int j, vector<vector<int>>& grid) {
        if (i < 0 || i >= grid.size() || j < 0 || j >= grid[i].size()) {
            return false;
        }

        if (grid[i][j] == 1) {
            return true;
        }

        grid[i][j] = 1;

        bool ans = true;

        for (int k = 0; k < 4; k++) {
            int x = i + m_dx[k];
            int y = j + m_dy[k];
            if (!this->dfs(x, y, grid)) {
                ans = false;
            }
        }

        return ans;
    }
};
```
