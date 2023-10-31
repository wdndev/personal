# 16.位运算

# 1.位运算符

-   [如何从十进制转换为二进制](https://zh.wikihow.com/从十进制转换为二进制 "如何从十进制转换为二进制")

机器里的数字表示方式和存储格式就是二进制

# 2.算数移位与逻辑移位

| 含义                  | 运算符 | 示例                  |
| ------------------- | --- | ------------------- |
| 左移                  | <<  | 0011 → 0110         |
| 右移                  | >>  | 0110 → 0011         |
| 按位或                 | \|  | 0011 \| 1011 → 1011 |
| 按位与                 | &   | 0011 & 1011→ 0011   |
| 按位取反                | \~  | 0011  → 1100        |
| 按位异或&#xA;(相同为0不同为1) | ^   | 0011 \| 1011 → 1000 |

# 3.位运算的应用

## 3.1 XOR - 异或

异或：相同为0，不同为1。也可用“不进位加法”来理解

异或的一些特点：

-   `x^0 = x`
-   `x^1s = ~x`  (注意 1s = \~0， “全1”)
-   `x^(~x) = 1s`
-   `x^x = 0`
-   `c = a ^ b → a^c = b, b^c = a` (交换两个数)
-   `a^b^c = a^(b^c)=(a^b)^c` (associative)

## 3.2 指定位置的位运算

1.  将x最右边的 n位清零: `x & (~0<< n)`
2.  获取x的第 n位值 (0或者 1) : `(x>> n) & 1`
3.  获取x的第 n位的幂值: `x & (1 <<(n-1))`
4.  仅将第n位置为 1: `x | (1 << n)`
5.  仅将第n位置为 0: `x & (~(1 << n)`
6.  将x最高位至第n位 (含) 清零: `x & ((1 << n)-1)`
7.  将第n位至第0位 (含) 清零: `x & (~((1<<(n +1))-1))`

## 3.3 实战位运算要点

-   判断奇偶
    -   `x % 2==1`  →  `(x & 1)==1`
    -   `x % 2==0`  →  `(x & 1)==0`
-   `x>>1` → `x/2`
    即: `x=x/2;`  -> `X=X >> 1;``mid =(left +right) / 2;` → `mid = (left +right) >> 1`
-   `X = X & (X - 1)` : 清零最低位的 1
-   `X & -X`  → 得到最低位的1
-   `X&~X → 0`

# 4.实战题目

## 4.1 位1的个数

[191. 位1的个数 - 力扣（LeetCode）](https://leetcode.cn/problems/number-of-1-bits/description/ "191. 位1的个数 - 力扣（LeetCode）")

```bash
编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为汉明重量）。

```

1.  for loop : 0 → 32
2.  %2, /2
3.  &1, x = x >> 1
4.  while (x > 0) { count ++; x = x & (x - 1) }

循环和位移动

```c++
// 1. 循环和位移动
// 遍历数字的32位，如果某一位为1，计数器加一
int hammingWeight1(uint32_t n) {
    int count = 0;
    int mask  = 1;
    for (int i = 0; i < 32; i++) {
        if ((n & mask) != 0) {
            count++;
        }

        mask <<= 1;
    }

    return count;
}
```

位操作

```c++
// 2.位操作
// 不断把数字最后一个1翻转，并计数+1.当数字变为0时，此时没有1了
// n 和 n-1 做与运算，会把最后一个1变为0
int hammingWeight(uint32_t n) {
    int count = 0;
    while (n != 0) {
        count++;
        n &= (n - 1);
    }

    return count;
}
```

## 4.2 2的幂

[231. 2 的幂 - 力扣（LeetCode）](https://leetcode.cn/problems/power-of-two/ "231. 2 的幂 - 力扣（LeetCode）")

```bash
给你一个整数 n，请你判断该整数是否是 2 的幂次方。如果是，返回 true ；否则，返回 false 。

如果存在一个整数 x 使得 n == 2^x ，则认为 n 是 2 的幂次方。
```

2的幂：**这个数的二进制表示，有且只有一个二进制位是1**

```c++
class Solution {
public:
    bool isPowerOfTwo(int n) {
        // 1.n > 0
        // 2.二进制表示只有一个1，打掉之后，肯定为0
        return (n > 0) && (n & (n - 1)) == 0;
    }
};
```

## 4.3 颠倒二进制位

[190. 颠倒二进制位 - 力扣（LeetCode）](https://leetcode.cn/problems/reverse-bits/description/ "190. 颠倒二进制位 - 力扣（LeetCode）")

```bash
颠倒给定的 32 位无符号整数的二进制位。

提示：

- 请注意，在某些语言（如 Java）中，没有无符号整数类型。在这种情况下，输入和输出都将被指定为有符号整数类型，并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
- 在 Java 中，编译器使用二进制补码记法来表示有符号整数。因此，在 示例 2 中，输入表示有符号整数 -3，输出表示有符号整数 -1073741825。
```

```c++
class Solution {
public:
    // 不断把n的最后一位输送到res的最后一位，res再不断的左移
    uint32_t reverseBits(uint32_t n) {
        uint32_t res = 0;
        // 操作32此移位操作
        int idx = 32;
        while (idx--) {
            // 结果左移一位，空出位置与n的最后一位相加
            res <<= 1;
            // 加上n的最后一位
            res += n & 1;
            // n右移一位，供下一轮与结果相加
            n >>= 1;
        }
        
        return res;
    }

    // 多次移位
    uint32_t reverseBits2(uint32_t n) {

        uint32_t res = 0;
        for ( int i = 0; i < 32; i++) {
            if (n & (1 << i)) {
                res |= 1 << (31 - i);
            }
        }
        
        return res;
    }
};
```

## 4.4 n皇后问题

[51. N 皇后 - 力扣（LeetCode）](https://leetcode.cn/problems/n-queens/ "51. N 皇后 - 力扣（LeetCode）")

[52. N 皇后 II - 力扣（LeetCode）](https://leetcode.cn/problems/n-queens-ii/description/ "52. N 皇后 II - 力扣（LeetCode）")

```bash
n 皇后问题 研究的是如何将 n 个皇后放置在 n × n 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 n ，返回 n 皇后问题 不同的解决方案的数量。
```

使用位运算解法

```python
class Solution:
    def totalNQueens(self, n: int) -> int:
        if n < 1:
            return 0
        self.count = 0
        self.dfs(n, 0, 0, 0, 0)
        return self.count
    
    def dfs(self, n, row, cols, pie, na):
        # 递归终止条件
        if row >= n:
            self.count += 1
            return
        
        # 得到当前所有的空位
        bits = (~(cols | pie | na)) & ((1 << n) - 1)

        while bits:
            # 取到最低为的1
            p = bits & -bits
            # 表示在p位置上放入皇后
            bits = bits & (bits - 1)
            self.dfs(n, row + 1, cols | p, (pie | p) << 1, (na | p) >> 1)
            # 不需要revert cols, pie, na 的状态
```

```c++
class Solution {
public:
    int totalNQueens(int n) {
        m_count = 0;
        m_size = (1 << n) - 1;
        this->dfs(0, 0, 0);
        return m_count;
    }
    
private:
    int m_size;
    int m_count;

    void dfs(int row, int pie, int na) {
        // 递归终止条件
        if (row == m_size) {
            m_count++;
            return;
        }
        // 得到当前所有空位
        int pos = m_size & (~(row | pie | na));

        while (pos != 0)
        {
            // 取到最低位的1
            int p = pos & (-pos);
            // 将p位置放入皇后
            pos -= p; // pos &= pos - 1
            this->dfs(row | p, (pie | p) << 1, (na | p) >> 1);
            // 不需要revert cols, pie, na 的状态
        }
        
    }
};
```

## 4.5 比特位计数（DP+）

[338. 比特位计数 - 力扣（LeetCode）](https://leetcode.cn/problems/counting-bits/description/ "338. 比特位计数 - 力扣（LeetCode）")

```bash
给你一个整数 n ，对于 0 <= i <= n 中的每个 i ，计算其二进制表示中 1 的个数 ，返回一个长度为 n + 1 的数组 ans 作为答案。

```

对于任意整数 xxx，令 `x=x & (x−1)`，该运算将 x 的二进制表示的最后一个 1 变成 0。因此，对 x 重复该操作，直到 x 变成 0，则操作次数即为 x 的「一比特数」。

```c++
class Solution {
public:
    int countOnes(int x) {
        int ones = 0;
        while (x > 0) {
            x &= (x - 1);
            ones++;
        }
        return ones;
    }

    vector<int> countBits(int n) {
        vector<int> bits(n + 1);
        for (int i = 0; i <= n; i++) {
            bits[i] = countOnes(i);
        }
        return bits;
    }
};

```
