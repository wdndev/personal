/*
 * @lc app=leetcode.cn id=705 lang=cpp
 *
 * [705] 设计哈希集合
 *
 * https://leetcode.cn/problems/design-hashset/description/
 *
 * algorithms
 * Easy (63.43%)
 * Likes:    322
 * Dislikes: 0
 * Total Accepted:    109.8K
 * Total Submissions: 173.1K
 * Testcase Example:  '["MyHashSet","add","add","contains","contains","add","contains","remove","contains"]\n' +
  '[[],[1],[2],[1],[3],[2],[2],[2],[2]]'
 *
 * 不使用任何内建的哈希表库设计一个哈希集合（HashSet）。
 * 
 * 实现 MyHashSet 类：
 * 
 * 
 * void add(key) 向哈希集合中插入值 key 。
 * bool contains(key) 返回哈希集合中是否存在这个值 key 。
 * void remove(key) 将给定值 key 从哈希集合中删除。如果哈希集合中没有这个值，什么也不做。
 * 
 * 
 * 
 * 示例：
 * 
 * 
 * 输入：
 * ["MyHashSet", "add", "add", "contains", "contains", "add", "contains",
 * "remove", "contains"]
 * [[], [1], [2], [1], [3], [2], [2], [2], [2]]
 * 输出：
 * [null, null, null, true, false, null, true, null, false]
 * 
 * 解释：
 * MyHashSet myHashSet = new MyHashSet();
 * myHashSet.add(1);      // set = [1]
 * myHashSet.add(2);      // set = [1, 2]
 * myHashSet.contains(1); // 返回 True
 * myHashSet.contains(3); // 返回 False ，（未找到）
 * myHashSet.add(2);      // set = [1, 2]
 * myHashSet.contains(2); // 返回 True
 * myHashSet.remove(2);   // set = [1]
 * myHashSet.contains(2); // 返回 False ，（已移除）
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 0 <= key <= 10^6
 * 最多调用 10^4 次 add、remove 和 contains
 * 
 * 
 */

// @lc code=start
class MyHashSet {
public:
    MyHashSet() {
        m_data.resize(m_base);
    }
    
    void add(int key) {
        int h = this->hash(key);
        for (auto it = m_data[h].begin(); it != m_data[h].end(); it++) {
            if ((*it) == key) {
                return;
            }
        }
        m_data[h].push_back(key);
    }
    
    void remove(int key) {
        int h = this->hash(key);
        for (auto it = m_data[h].begin(); it != m_data[h].end(); it++) {
            if ((*it) == key) {
                m_data[h].erase(it);
                return;
            }
        }
    }
    
    bool contains(int key) {
        int h = this->hash(key);
        for (auto it = m_data[h].begin(); it != m_data[h].end(); it++) {
            if ((*it) == key) {
                return true; 
            }
        }
        return false;
    }
private:
    std::vector<std::list<int>> m_data;
    static const int m_base = 769;
    static int hash(int key) {
        return key % m_base;
    }
};

/**
 * Your MyHashSet object will be instantiated and called as such:
 * MyHashSet* obj = new MyHashSet();
 * obj->add(key);
 * obj->remove(key);
 * bool param_3 = obj->contains(key);
 */
// @lc code=end

