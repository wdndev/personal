// https://leetcode.cn/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/description/

// 闯关游戏需要破解一组密码，闯关组给出的有关密码的线索是：

// - 一个拥有密码所有元素的非负整数数组 password
// - 密码是 password 中所有元素拼接后得到的最小的一个数

// 请编写一个程序返回这个密码。

// 示例 1:

// 输入: password = [15, 8, 7]
// 输出: "1578"

class Solution {
public:
    string crackPassword(vector<int>& password) {
        std::vector<std::string> strs;
        std::string ans;
        // 变换为字符串
        for (auto& pw : password) {
            strs.push_back(std::to_string(pw));
        }
        // 自定义排序
        std::sort(strs.begin(), strs.end(), [](std::string& x, std::string& y){
            return x + y < y + x;
        });
        // 组成答案
        for (auto& s : strs) {
            ans.append(s);
        }
        return ans;
    }
};



