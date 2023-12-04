# 12 Stack

# 1.有效括号

[20. 有效的括号 - 力扣（LeetCode）](https://leetcode.cn/problems/valid-parentheses/description/?envType=study-plan-v2\&envId=top-100-liked "20. 有效的括号 - 力扣（LeetCode）")

```bash
给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串 s ，判断字符串是否有效。

有效字符串需满足：

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
每个右括号都有一个对应的相同类型的左括号。
 

示例 1：

输入：s = "()"
输出：true
```

1.  栈

```c++
class Solution {
public:
    bool isValid(string s) {
        std::stack<char> stk;

        for (int i = 0; i < s.size(); i++) {
            // 左括号，入栈
            if (s[i] == '(' || s[i] == '[' || s[i] == '{') {
                stk.push(s[i]);
            } else {
            // 右括号处理
                // 如果栈为空，则直接返回
                if (stk.empty()) {
                    return false;
                }

                // 出栈，比对括号
                char tmp_char = stk.top();
                stk.pop();
                if (s[i] == ')' && tmp_char != '(') {
                    return false;
                }
                if (s[i] == ']' && tmp_char != '[') {
                    return false;
                }
                if (s[i] == '}' && tmp_char != '{') {
                    return false;
                }

            }
        }

        // 当所有元素遍历结束，栈空，则说明匹配成功
        return stk.empty();
    }
};
```

# 2.最小栈

[155. 最小栈 - 力扣（LeetCode）](https://leetcode.cn/problems/min-stack/description/?envType=study-plan-v2\&envId=top-100-liked "155. 最小栈 - 力扣（LeetCode）")

```bash
设计一个支持 push ，pop ，top 操作，并能在常数时间内检索到最小元素的栈。

实现 MinStack 类:

- MinStack() 初始化堆栈对象。
- void push(int val) 将元素val推入堆栈。
- void pop() 删除堆栈顶部的元素。
- int top() 获取堆栈顶部的元素。
- int getMin() 获取堆栈中的最小元素。

示例 1:

输入：
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

输出：
[null,null,null,null,-3,null,0,-2]

解释：
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.getMin();   --> 返回 -2.

```

辅助栈：每个元素 `a` 入栈时把当前栈的最小值 `m` 存储起来。在这之后无论何时，如果栈顶元素是 `a`，我们就可以直接返回存储的最小值 `m`。

可以使用一个辅助栈，与元素栈同步插入与删除，用于存储与每个元素对应的最小值。

-   当一个元素要入栈时，取当前辅助栈的栈顶存储的最小值，与当前元素比较得出最小值，将这个最小值插入辅助栈中；
-   当一个元素要出栈时，我们把辅助栈的栈顶元素也一并弹出；
-   在任意一个时刻，栈内元素的最小值就存储在辅助栈的栈顶元素中。

```c++
class MinStack {
public:
    MinStack() {
        m_min_stack.push(m_min);
    }
    
    void push(int val) {
        m_stack.push(val);
        m_min = std::min(m_min, val);
        m_min_stack.push(m_min);
    }
    
    void pop() {
        m_stack.pop();
        m_min_stack.pop();
        m_min = m_min_stack.top();
    }
    
    int top() {
        return m_stack.top();
    }
    
    int getMin() {
        return m_min;
    }

private:
    std::stack<int> m_stack;
    std::stack<int> m_min_stack;
    int m_min = INT_MAX;
};
```

# 3.字符串解码

[394. 字符串解码 - 力扣（LeetCode）](https://leetcode.cn/problems/decode-string/description/?envType=study-plan-v2\&envId=top-100-liked "394. 字符串解码 - 力扣（LeetCode）")

```bash
给定一个经过编码的字符串，返回它解码后的字符串。

编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。

你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。

此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。

 

示例 1：

输入：s = "3[a]2[bc]"
输出："aaabcbc"
示例 2：

输入：s = "3[a2[c]]"
输出："accaccacc"
```

数字只表示重复次数， 中括号可作为分界线，解码应该从最内部的中括号开始

栈：左括号和所有数字（计算出值）字母全放进去

碰到右括号，依次取出栈顶元素，放入字符串中，直到取出左括号为止，用栈顶元素翻倍所有已取出的字母，并且取出栈顶的数字，把字符串放入栈中

直到读取到末尾字母，怎么判断是不是数字，左括号前面肯定是数字

```c++
class Solution {
public:
    string decodeString(string s) {
        std::stack<char> stk;
        std::string tmp_str = "";
        for (int i = 0; i < s.size(); i++) {
            if (s[i] != ']') {
                stk.push(s[i]);
            } else {

                // 取出字符
                tmp_str = "";
                while (!stk.empty() && stk.top() != '[') {
                    tmp_str = tmp_str + stk.top();
                    stk.pop();
                }
                // 弹出 [
                stk.pop();

                // 取出数字
                int num = 0;    // 表示读取到的数字
                int n = 0;      // 表示第几位数字
                while (!stk.empty() && isdigit(stk.top())) {
                    int po = pow(10, n);
                    num = (stk.top() - '0') * po + num;
                    n++;

                    stk.pop();
                }

                // 把num个tmp_str放入栈中
                for (int j = 0; j < num; j++) {
                    for (int k = tmp_str.size() - 1; k >= 0 ; k--) {
                        stk.push(tmp_str[k]);
                    }
                }
                
            }
        }

        std::string ans;
        while (!stk.empty()) {
            char tmp_char = stk.top();
            stk.pop();
            ans = tmp_char + ans;
        }

        return ans;
    }
};
```

# 4.每日温度

[739. 每日温度 - 力扣（LeetCode）](https://leetcode.cn/problems/daily-temperatures/description/?envType=study-plan-v2\&envId=top-100-liked "739. 每日温度 - 力扣（LeetCode）")

```bash
给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指对于第 i 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 0 来代替。


示例 1:

输入: temperatures = [73,74,75,71,69,72,76,73]
输出: [1,1,4,2,1,1,0,0]
示例 2:

输入: temperatures = [30,40,50,60]
输出: [1,1,1,0]
```

1.  暴力

可以维护一个数组 `next `记录每个温度第一次出现的下标。数组 `next `中的元素初始化为无穷大，在遍历温度列表的过程中更新 `next `的值。

反向遍历温度列表。对于每个元素 `temperatures[i]`，在数组 `next `中找到从 `temperatures[i] + 1` 到 `100 `中每个温度第一次出现的下标，将其中的最小下标记为 `warmer_index`，则 `warmer_index `为下一次温度比当天高的下标。如果 `warmer_index `不为无穷大，则 `warmerIndex - i` 即为下一次温度比当天高的等待天数，最后令 `next[temperatures[i]] = i`。

1.  单调栈：可以维护一个存储下标的单调栈，从栈底到栈顶的下标对应的温度列表中的温度依次递减。如果一个下标在单调栈里，则表示尚未找到下一次温度更高的下标。

正向遍历温度列表。对于温度列表中的每个元素 `temperatures[i]`，如果栈为空，则直接将 `i` 进栈，如果栈不为空，则比较栈顶元素 `prevIndex `对应的温度 `temperatures[prevIndex]` 和当前温度 `temperatures[i]`，如果 `temperatures[i] > temperatures[prevIndex]`，则将 `prevIndex `移除，并将 `prevIndex `对应的等待天数赋为` i - prevIndex`，重复上述操作直到栈为空或者栈顶元素对应的温度小于等于当前温度，然后将 `i` 进栈。

```c++
class Solution {
public:
    // 1.暴力解法
    vector<int> dailyTemperatures1(vector<int>& temperatures) {
        int n = temperatures.size();
        std::vector<int> ans(n);
        // 记录每个温度第一次出现的下标
        std::vector<int> next(101, INT_MAX);

        // 反向遍历
        for (int i = n - 1; i >= 0; i--) {
            int warmer_idx = INT_MAX;
            // 对于每个元素 temperatures[i]，在数组 next 中找到从 
            // temperatures[i] + 1 到 100 中每个温度第一次出现的下标
            // warmer_index 为下一次温度比当天高的下标
            for (int t = temperatures[i] + 1; t <= 100; t++) {
                warmer_idx = std::min(warmer_idx, next[t]);
            }

            if (warmer_idx != INT_MAX) {
                ans[i] = warmer_idx - i;
            }

            next[temperatures[i]] = i;
        }

        return ans;
    }

    // 2.单调栈
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        int n = temperatures.size();
        std::vector<int> ans(n);
        std::stack<int> stk;

        for (int i = 0; i < n; i++) {
            while (!stk.empty() && temperatures[i] > temperatures[stk.top()]) {
                int previous_idx = stk.top();
                ans[previous_idx] = i - previous_idx;
                stk.pop();
            }
            stk.push(i);
        }

        return ans;
    }
};
```

# 5.柱状图中最大的矩形

[84. 柱状图中最大的矩形 - 力扣（LeetCode）](https://leetcode.cn/problems/largest-rectangle-in-histogram/description/ "84. 柱状图中最大的矩形 - 力扣（LeetCode）")

```bash
给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。

```

```c++
class Solution {
public:
    // 1.固定高度，枚举宽度， 超时
    // 固定每个数组元素为每个矩形的高，然后遍历数组，寻找每个矩形高能构成的最大面积，
    // 当左右两边第一次出现比当前高小的元素值，即为当前高能构成的最大值，每次保存最大值
    int largestRectangleArea1(vector<int>& heights) {
        int max_area = 0;
        // 遍历高度
        for (int mid = 0; mid < heights.size(); mid++) {
            int h = heights[mid];
            int left = mid;
            int right = mid;
            // 左侧寻找最大宽度 
            while (left - 1 >= 0 && heights[left - 1] >= h) {
                left--;
            }
            // 右侧寻找最大宽度
            while (right + 1 < heights.size() && heights[right + 1] >= h) {
                right++;
            }

            max_area = std::max(max_area, h * (right - left + 1));
        }
        
        return max_area;
    }

    // 2.固定宽度，枚举高度, 超时
    // 固定左右两边的长度即固定宽的长度，然后遍历数组，寻找当前长度中高最短的元素，
    // 即当前宽能构成的最大矩形，每次保存最大值
    int largestRectangleArea2(vector<int>& heights) {
        int max_area = 0;
        int n = heights.size();
        if (n == 1) {
            return heights[0];
        }
        for (int left = 0; left < n; left++) {
            int min_height = heights[left];
            for (int right = left; right < n; right++) {
                min_height = std::min(min_height, heights[right]);
                max_area = std::max(max_area, min_height * (right - left + 1));
            }
            
        }

        return max_area;
    }

    // 3.单调栈
    // 在枚举宽的同时需要寻找高，在枚举高的时候又要寻找宽，时间消耗非常大
    // 那么可以利用递增栈优化暴力暴力求解的过程

    // 当元素大于栈顶元素时，入栈
    // 当元素小于栈顶元素时，维护栈的递增性，将小于当前元素的栈顶元素弹出，并计算面积
    int largestRectangleArea(vector<int>& heights) {
        int n = heights.size();
        if (n == 1) {
            return heights[0];
        }
        
        int max_area = 0;

        std::stack<int> stack;
        // 遍历数组
        for (int i = 0; i < n; i++) {
            while (!stack.empty() && heights[stack.top()] >= heights[i]) {
                // 出栈，并计算面积，维护递增性，需要对小于的元素全部出栈
                int length = heights[stack.top()];
                stack.pop();

                int weight = i;
                // 最后一个栈顶元素，出栈计算面积需要包含一下前面和后面，
                // 因为矩形可以延伸，这里需要好好想一想
                if (!stack.empty()) {
                    weight = i - stack.top() - 1;
                }

                max_area = std::max(max_area, length * weight);

            }
            // 入栈
            stack.push(i);
        }

        // 数组元素全部遍历完了，但是栈还有元素，进行清空栈
        while (!stack.empty()) {
            int length = heights[stack.top()];
            stack.pop();
            int weight = n;
            if (!stack.empty()) {
                weight = n - stack.top() - 1;
            }
            max_area = std::max(max_area, length * weight);
        }

        return max_area;
    }
};
```
