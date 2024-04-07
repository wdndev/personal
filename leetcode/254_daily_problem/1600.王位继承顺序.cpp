/*
 * @lc app=leetcode.cn id=1600 lang=cpp
 *
 * [1600] 王位继承顺序
 *
 * https://leetcode.cn/problems/throne-inheritance/description/
 *
 * algorithms
 * Medium (65.72%)
 * Likes:    106
 * Dislikes: 0
 * Total Accepted:    31.3K
 * Total Submissions: 45.4K
 * Testcase Example:  '["ThroneInheritance","birth","birth","birth","birth","birth","birth","getInheritanceOrder","death","getInheritanceOrder"]\n' +
  '[["king"],["king","andy"],["king","bob"],["king","catherine"],["andy","matthew"],["bob","alex"],["bob","asha"],[null],["bob"],[null]]'
 *
 * 一个王国里住着国王、他的孩子们、他的孙子们等等。每一个时间点，这个家庭里有人出生也有人死亡。
 * 
 * 这个王国有一个明确规定的王位继承顺序，第一继承人总是国王自己。我们定义递归函数 Successor(x, curOrder) ，给定一个人 x
 * 和当前的继承顺序，该函数返回 x 的下一继承人。
 * 
 * 
 * Successor(x, curOrder):
 * ⁠   如果 x 没有孩子或者所有 x 的孩子都在 curOrder 中：
 * ⁠       如果 x 是国王，那么返回 null
 * ⁠       否则，返回 Successor(x 的父亲, curOrder)
 * ⁠   否则，返回 x 不在 curOrder 中最年长的孩子
 * 
 * 
 * 比方说，假设王国由国王，他的孩子 Alice 和 Bob （Alice 比 Bob 年长）和 Alice 的孩子 Jack 组成。
 * 
 * 
 * 一开始， curOrder 为 ["king"].
 * 调用 Successor(king, curOrder) ，返回 Alice ，所以我们将 Alice 放入 curOrder 中，得到
 * ["king", "Alice"] 。
 * 调用 Successor(Alice, curOrder) ，返回 Jack ，所以我们将 Jack 放入 curOrder 中，得到 ["king",
 * "Alice", "Jack"] 。
 * 调用 Successor(Jack, curOrder) ，返回 Bob ，所以我们将 Bob 放入 curOrder 中，得到 ["king",
 * "Alice", "Jack", "Bob"] 。
 * 调用 Successor(Bob, curOrder) ，返回 null 。最终得到继承顺序为 ["king", "Alice", "Jack",
 * "Bob"] 。
 * 
 * 
 * 通过以上的函数，我们总是能得到一个唯一的继承顺序。
 * 
 * 请你实现 ThroneInheritance 类：
 * 
 * 
 * ThroneInheritance(string kingName) 初始化一个 ThroneInheritance
 * 类的对象。国王的名字作为构造函数的参数传入。
 * void birth(string parentName, string childName) 表示 parentName 新拥有了一个名为
 * childName 的孩子。
 * void death(string name) 表示名为 name 的人死亡。一个人的死亡不会影响 Successor
 * 函数，也不会影响当前的继承顺序。你可以只将这个人标记为死亡状态。
 * string[] getInheritanceOrder() 返回 除去 死亡人员的当前继承顺序列表。
 * 
 * 
 * 
 * 
 * 示例：
 * 
 * 
 * 输入：
 * ["ThroneInheritance", "birth", "birth", "birth", "birth", "birth", "birth",
 * "getInheritanceOrder", "death", "getInheritanceOrder"]
 * [["king"], ["king", "andy"], ["king", "bob"], ["king", "catherine"],
 * ["andy", "matthew"], ["bob", "alex"], ["bob", "asha"], [null], ["bob"],
 * [null]]
 * 输出：
 * [null, null, null, null, null, null, null, ["king", "andy", "matthew",
 * "bob", "alex", "asha", "catherine"], null, ["king", "andy", "matthew",
 * "alex", "asha", "catherine"]]
 * 
 * 解释：
 * ThroneInheritance t= new ThroneInheritance("king"); // 继承顺序：king
 * t.birth("king", "andy"); // 继承顺序：king > andy
 * t.birth("king", "bob"); // 继承顺序：king > andy > bob
 * t.birth("king", "catherine"); // 继承顺序：king > andy > bob > catherine
 * t.birth("andy", "matthew"); // 继承顺序：king > andy > matthew > bob > catherine
 * t.birth("bob", "alex"); // 继承顺序：king > andy > matthew > bob > alex >
 * catherine
 * t.birth("bob", "asha"); // 继承顺序：king > andy > matthew > bob > alex > asha >
 * catherine
 * t.getInheritanceOrder(); // 返回 ["king", "andy", "matthew", "bob", "alex",
 * "asha", "catherine"]
 * t.death("bob"); // 继承顺序：king > andy > matthew > bob（已经去世）> alex > asha >
 * catherine
 * t.getInheritanceOrder(); // 返回 ["king", "andy", "matthew", "alex", "asha",
 * "catherine"]
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= kingName.length, parentName.length, childName.length, name.length <=
 * 15
 * kingName，parentName， childName 和 name 仅包含小写英文字母。
 * 所有的参数 childName 和 kingName 互不相同。
 * 所有 death 函数中的死亡名字 name 要么是国王，要么是已经出生了的人员名字。
 * 每次调用 birth(parentName, childName) 时，测试用例都保证 parentName 对应的人员是活着的。
 * 最多调用 10^5 次birth 和 death 。
 * 最多调用 10 次 getInheritanceOrder 。
 * 
 * 
 */

// @lc code=start
class ThroneInheritance {
public:
    // 题目中定义的 Successor(x, curOrder)函数，与多叉树的前序遍历过程是一致的：
    // 「返回 x 不在 curOrder 中最年长的孩子」对应着选择 x 在树中的一个子节点，递归地进行遍历操作；
    // 「返回 Successor(x 的父亲, curOrder)」对应着当我们将以 x为根的子树遍历完成后，回溯到 x 的父节点继续进行遍历；
    // 「返回 null」对应着我们将整棵树遍历完成。
    ThroneInheritance(string kingName) {
        m_king_name = kingName;
    }
    
    void birth(string parentName, string childName) {
        // 将孩子添加到列表中
        m_children[parentName].push_back(childName);
    }
    
    void death(string name) {
        m_death.insert(name);
    }
    
    vector<string> getInheritanceOrder() {
        std::vector<std::string> ans;
        // 从国王开始遍历
        this->preorder(m_king_name, ans);

        return ans;
    }
private:
    // 国王名字
    std::string m_king_name;
    // 孩子列表，key是父亲，value为孩子
    std::unordered_map<std::string, std::vector<std::string>> m_children;
    // 死亡集合
    std::unordered_set<std::string> m_death;

    void preorder(const std::string& name, std::vector<std::string>& ans) {
        // 如果还活着，添加带继承列表中
        if (!m_death.count(name)) {
            ans.push_back(name);
        }
        // 如果有孩子，添加孩子
        if (m_children.find(name) != m_children.end()) {
            // DFS遍历所有孩子
            for (auto& child_name : m_children[name]) {
                this->preorder(child_name, ans);
            }
        }
    }
};


/**
 * Your ThroneInheritance object will be instantiated and called as such:
 * ThroneInheritance* obj = new ThroneInheritance(kingName);
 * obj->birth(parentName,childName);
 * obj->death(name);
 * vector<string> param_3 = obj->getInheritanceOrder();
 */
// @lc code=end

