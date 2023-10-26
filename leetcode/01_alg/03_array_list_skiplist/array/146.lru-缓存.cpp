/*
 * @lc app=leetcode.cn id=146 lang=cpp
 *
 * [146] LRU 缓存
 *
 * https://leetcode.cn/problems/lru-cache/description/
 *
 * algorithms
 * Medium (53.76%)
 * Likes:    2983
 * Dislikes: 0
 * Total Accepted:    550.6K
 * Total Submissions: 1M
 * Testcase Example:  '["LRUCache","put","put","get","put","get","put","get","get","get"]\n' +
  '[[2],[1,1],[2,2],[1],[3,3],[2],[4,4],[1],[3],[4]]'
 *
 * 请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
 * 
 * 实现 LRUCache 类：
 * 
 * 
 * 
 * 
 * LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
 * int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
 * void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组
 * key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。
 * 
 * 
 * 函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。
 * 
 * 
 * 
 * 
 * 
 * 示例：
 * 
 * 
 * 输入
 * ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
 * [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
 * 输出
 * [null, null, null, 1, null, -1, null, -1, 3, 4]
 * 
 * 解释
 * LRUCache lRUCache = new LRUCache(2);
 * lRUCache.put(1, 1); // 缓存是 {1=1}
 * lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
 * lRUCache.get(1);    // 返回 1
 * lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
 * lRUCache.get(2);    // 返回 -1 (未找到)
 * lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
 * lRUCache.get(1);    // 返回 -1 (未找到)
 * lRUCache.get(3);    // 返回 3
 * lRUCache.get(4);    // 返回 4
 * 
 * 
 * 
 * 
 * 提示：
 * 
 * 
 * 1 <= capacity <= 3000
 * 0 <= key <= 10000
 * 0 <= value <= 10^5
 * 最多调用 2 * 10^5 次 get 和 put
 * 
 * 
 */

// @lc code=start
class LRUCache {
public:
    using Pair = std::pair<int, int>;
    using List = std::list<Pair>;
    using Map = std::unordered_map<int, typename List::iterator>;

    LRUCache(int capacity) {
        m_capacity = capacity;
    }
    
    int get(int key) {
        // 查找key是否在哈希表中
        typename Map::iterator map_itor = m_map.find(key);
        // 不存在，返回-1
        if (map_itor == m_map.end())
            return -1;

        // 如果存在
        // 1.链表要将数据删除
        // 2.在将数据加入到链表队头
        // 目的是为了维护链表队头的hi最近访问的数据

        // 取出哈希表的value值，也就是链表节点
        typename List::iterator list_itor = map_itor->second;

        // 创建新的键值对
        std::pair<int, int> list_pair = std::make_pair(list_itor->first, list_itor->second);

        // 从链表中删除该节点
        m_list.erase(list_itor);

        // 将数据加入到队头
        m_list.push_front(list_pair);

        // 更新哈希表
        m_map[key] = m_list.begin();

        return list_pair.second;
    }
    
    void put(int key, int value) {
        // 查找key是否在哈希表中
        typename Map::iterator itor = m_map.find(key);
        // 如果存在，则要将老数据从哈希表和链表中移除
        if (itor != m_map.end())
        {
            m_list.erase(itor->second);
            m_map.erase(itor);
        }

        // 插入到链表头
        m_list.push_front(std::make_pair(key, value));
        // 将链表头放入hash表中
        m_map[key] = m_list.begin();

        // 当链表大小超过阈值后，删除
        if (m_list.size() > m_capacity)
        {
            int end_key = m_list.back().first;
            m_list.pop_back();
            m_map.erase(end_key);
        }
    }
private:
    int m_capacity;
    List m_list;
    Map m_map;
};

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache* obj = new LRUCache(capacity);
 * int param_1 = obj->get(key);
 * obj->put(key,value);
 */
// @lc code=end

