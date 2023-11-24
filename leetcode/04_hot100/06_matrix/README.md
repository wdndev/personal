# 06 matrix

# 1.矩阵置零

[73. 矩阵置零 - 力扣（LeetCode）](https://leetcode.cn/problems/set-matrix-zeroes/description/?envType=study-plan-v2\&envId=top-100-liked "73. 矩阵置零 - 力扣（LeetCode）")

```bash
给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。


```

两次遍历即可

```c++
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int rows = matrix.size();
        int cols = matrix[0].size();
        // 行列标记数据，用于标记是否存在0
        std::vector<int> row_flag(rows), col_flag(cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (matrix[i][j] == 0)
                {
                    row_flag[i] = true;
                    col_flag[j] = true;
                }
            }
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (row_flag[i] || col_flag[j]) {
                    matrix[i][j] = 0;
                }
            }
        }
    }
};
```

# 2.螺旋矩阵

[54. 螺旋矩阵 - 力扣（LeetCode）](https://leetcode.cn/problems/spiral-matrix/description/?envType=study-plan-v2\&envId=top-100-liked "54. 螺旋矩阵 - 力扣（LeetCode）")

```bash
给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。

```

**按层模拟**：可以将矩阵看成若干层，首先输出最外层的元素，其次输出次外层的元素，直到输出最内层的元素。

对于每层，从左上方开始以顺时针的顺序遍历所有元素。假设当前层的左上角位于 (top,left)，右下角位于 (bottom,right)，按照如下顺序遍历当前层的元素。

1.  从左到右遍历上侧元素，依次为 (top,left) 到 (top,right)。
2.  从上到下遍历右侧元素，依次为 (top+1,right)到 (bottom,right)。
3.  如果 left\<right且 top\<bottom，则从右到左遍历下侧元素，依次为 (bottom,right−1) 到 (bottom,left+1)，以及从下到上遍历左侧元素，依次为 (bottom,left) 到 (top+1,left)。

遍历完当前层的元素之后，将 left和 top 分别增加 1，将 right 和 bottom分别减少 1，进入下一层继续遍历，直到遍历完所有元素为止。

```c++
class Solution {
public:
    // 按层模拟：可以将矩阵看成若干层，首先输出最外层的元素，
    // 其次输出次外层的元素，直到输出最内层的元素。
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        if (matrix.size() == 0 || matrix[0].size() == 0) {
            return {};
        }

        int rows = matrix.size();
        int cols = matrix[0].size();
        std::vector<int> order;

        int left = 0;
        int right = cols - 1;
        int top = 0;
        int bottom = rows - 1;

        while (left <= right && top <= bottom) {
            // 从左到右
            for (int col = left; col <= right; col++) {
                order.push_back(matrix[top][col]);
            }
            // 从上到下
            for (int row = top + 1; row <= bottom; row++) {
                order.push_back(matrix[row][right]);
            }
            // 现在idx在右下角
            if (left < right && top < bottom) {
                // 从右到左
                for (int col = right - 1; col > left; col--) {
                    order.push_back(matrix[bottom][col]);
                }
                // 从下到上
                for (int row = bottom; row > top; row--) {
                    order.push_back(matrix[row][left]);
                }
            }

            left++;
            right--;
            top++;
            bottom--;
        }

        return order;
    }
};
```

# 3.旋转图像

[48. 旋转图像 - 力扣（LeetCode）](https://leetcode.cn/problems/rotate-image/description/?envType=study-plan-v2\&envId=top-100-liked "48. 旋转图像 - 力扣（LeetCode）")

```bash
给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。

你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。
```

由于矩阵中的行列从 0 开始计数，因此对于矩阵中的元素 `matrix[row][col]`，在旋转后，它的新位置为 `matrixnew[col][n−row−1]`

使用临时变量交换四个元素位置

$$
\left\{\begin{array}{ll}\operatorname{temp} & =\text { matrix }[\operatorname{row}][\operatorname{col}] \\ \operatorname{matrix}[\operatorname{row}][\text { col }] & =\operatorname{matrix}[n-\operatorname{col}-1][\operatorname{row}] \\ \operatorname{matrix}[n-\operatorname{col}-1][\operatorname{row}] & =\operatorname{matrix}[n-\operatorname{row}-1][n-\operatorname{col}-1] \\ \operatorname{matrix}[n-\operatorname{row}-1][n-\operatorname{col}-1] & =\operatorname{matrix}[\operatorname{col}][n-\operatorname{row}-1] \\ \operatorname{matrix}[\operatorname{col}][n-\operatorname{row}-1] & =\text { temp }\end{array}\right.
$$

```c++
class Solution {
public:
    // matrix[i][j]←matrix[n−1−j][i]←matrix[n−1−i][n−1−j]←matrix[j][n−1−i]←tmp
    // 当矩阵大小 n 为偶数时，取前 n/2 行、前 n/2列的元素为起始点；
    // 当矩阵大小 n 为奇数时，取前 n/2 行、前 (n+1)/2列的元素为起始点

    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();

        for (int i = 0; i < n / 2; i++)
        {
            for (int j = 0; j < (n+1) / 2; j++)
            {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[n -j - 1][i];
                matrix[n - j -1][i] = matrix[n - i - 1][n - j - 1];
                matrix[n - i - 1][n - j - 1] = matrix[j][n - i - 1];
                matrix[j][n - i - 1] = tmp;
            }
        }

    }
};
```

# 4.搜索二维矩阵Ⅱ

[240. 搜索二维矩阵 II - 力扣（LeetCode）](https://leetcode.cn/problems/search-a-2d-matrix-ii/description/?envType=study-plan-v2\&envId=top-100-liked "240. 搜索二维矩阵 II - 力扣（LeetCode）")

```bash
编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：

每行的元素从左到右升序排列。
每列的元素从上到下升序排列。
```

1.  暴力求解
2.  二分查找

```c++
class Solution {
public:
    // 1.暴力查找
    bool searchMatrix1(vector<vector<int>>& matrix, int target) {
        for (const auto& row : matrix) {
            for (auto& elem : row) {
                if (elem == target) {
                    return true;
                }
            }
        }

        return false;
    }

    // 2.二分查找, 遍历行/列，然后再对列/行进行二分
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        for (const auto& row : matrix) {
            int left = 0;
            int right = row.size() - 1;
            while (left < right) {
                // int mid = left + (right - left) / 2;
                int mid = left + right + 1 >> 1;
                if (row[mid] <= target) {
                    left = mid;
                } else {
                    right = mid - 1;
                }
            }
            if (row[right] == target) {
                return true;
            }
        }

        return false;
    }
};
```
