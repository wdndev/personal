/*
 * @lc app=leetcode.cn id=2288 lang=cpp
 *
 * [2288] 价格减免
 *
 * https://leetcode.cn/problems/apply-discount-to-prices/description/
 *
 * algorithms
 * Medium (32.02%)
 * Likes:    34
 * Dislikes: 0
 * Total Accepted:    21.9K
 * Total Submissions: 53.4K
 * Testcase Example:  '"there are $1 $2 and 5$ candies in the shop"\n50'
 *
 * 句子 是由若干个单词组成的字符串，单词之间用单个空格分隔，其中每个单词可以包含数字、小写字母、和美元符号 '$'
 * 。如果单词的形式为美元符号后跟着一个非负实数，那么这个单词就表示一个 价格 。
 * 
 * 
 * 例如 "$100"、"$23" 和 "$6" 表示价格，而 "100"、"$" 和 "$1e5 不是。
 * 
 * 
 * 给你一个字符串 sentence 表示一个句子和一个整数 discount 。对于每个表示价格的单词，都在价格的基础上减免 discount% ，并
 * 更新 该单词到句子中。所有更新后的价格应该表示为一个 恰好保留小数点后两位 的数字。
 * 
 * 返回表示修改后句子的字符串。
 * 
 * 注意：所有价格 最多 为  10 位数字。
 * 
 * 
 * 
 * 示例 1：
 * 
 * 
 * 输入：sentence = "there are $1 $2 and 5$ candies in the shop", discount = 50
 * 输出："there are $0.50 $1.00 and 5$ candies in the shop"
 * 解释：
 * 表示价格的单词是 "$1" 和 "$2" 。 
 * - "$1" 减免 50% 为 "$0.50" ，所以 "$1" 替换为 "$0.50" 。
 * - "$2" 减免 50% 为 "$1" ，所以 "$1" 替换为 "$1.00" 。
 * 
 * 示例 2：
 * 
 * 
 * 输入：sentence = "1 2 $3 4 $5 $6 7 8$ $9 $10$", discount = 100
 * 输出："1 2 $0.00 4 $0.00 $0.00 7 8$ $0.00 $10$"
 * 解释：
 * 任何价格减免 100% 都会得到 0 。
 * 表示价格的单词分别是 "$3"、"$5"、"$6" 和 "$9"。
 * 每个单词都替换为 "$0.00"。
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= sentence.length <= 10^5
 * sentence 由小写英文字母、数字、' ' 和 '$' 组成
 * sentence 不含前导和尾随空格
 * sentence 的所有单词都用单个空格分隔
 * 所有价格都是 正 整数且不含前导零
 * 所有价格 最多 为  10 位数字
 * 0 <= discount <= 100
 * 
 * 
 */

// @lc code=start
class Solution {
public:
    string discountPrices(string sentence, int discount) {
        double d = 1 - discount / 100.0;
        std::stringstream ss(sentence);
        std::string ans;
        std::string word;
        while (ss >> word) {
            ans += this->decide(word, discount);
            ans += " ";
        }
        ans.pop_back();
        return ans;
    }

    std::string decide(const std::string& word, int discount) {
        if (word[0] != '$' || word.size() == 1) {
            return word;
        }

        for (int i = 1; i < word.size(); i++) {
            if (word[i] < '0' || word[i] > '9') {
                return word;
            }
        }
        double ans = 0;
        for (int i = 1; i < word.size(); i++) {
            ans *= 10;
            int tmp = (word[i] - '0');
            ans += tmp;
        }

        //然后查看应该多少
        ans = ans * ((100 - discount)/100.0);
        stringstream ss;
        ss << std::fixed << std::setprecision(2) << ans;
        return "$" + ss.str();
    }
};
// @lc code=end

