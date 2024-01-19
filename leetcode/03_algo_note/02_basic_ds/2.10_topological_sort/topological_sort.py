""" 拓扑排序
"""

import collections

class TopologicalSort:
    """ 拓扑排序
    """
    def topological_sort_kahn(self, graph:dict):
        """ Kahn 算法实现拓扑排序，graph中包含所有顶点的有向边关系
            1. 不断找寻有向图中入度为 $0$ 的顶点，将其输出。
            2. 然后删除入度为 $0$ 的顶点和从该顶点出发的有向边。
            3. 重复上述操作直到图为空，或者找不到入度为 $0$ 的节点为止。
        """
        # indegrees用于记录所有顶点的入度
        indegrees = {u : 0 for u in graph}
        for u in graph:
            for v in graph[u]:
                indegrees[v] += 1

        # 将入度为0的顶点存入集合S中
        S = collections.deque([u for u in indegrees if indegrees[u] == 0])
        # order用于存储拓扑排序序列
        order = []

        while S:
            u = S.pop()                     # 从集合中选择一个没有前驱的顶点 0
            order.append(u)                 # 将其输出到拓扑序列 order 中
            for v in graph[u]:              # 遍历顶点 u 的邻接顶点 v
                indegrees[v] -= 1           # 删除从顶点 u 出发的有向边
                if indegrees[v] == 0:       # 如果删除该边后顶点 v 的入度变为 0
                    S.append(v)             # 将其放入集合 S 中
        
        if len(indegrees) != len(order):    # 还有顶点未遍历（存在环），无法构成拓扑序列
            return []
        return order                        # 返回拓扑序列
    
    def topological_sort_dfs(self, graph:dict):
        """ DFS实现拓扑排序，graph中包含所有顶点的有向边关系
            1. 对于一个顶点$u$，深度优先遍历从该顶点出发的有向边$<u,v>$。
               如果从该顶点 $u$ 出发的所有相邻顶点 $v$ 都已经搜索完毕，则回溯到顶点 $u$ 时，
               该顶点 $u$ 应该位于其所有相邻顶点 $v$ 的前面（拓扑序列中）。
            2. 这样一来，当对每个顶点进行深度优先搜索，在回溯到该顶点时将其放入栈中，
               则最终从栈顶到栈底的序列就是一种拓扑排序。
        """
        visited = set()                     # 记录当前顶点是否被访问过
        on_stack = set()                    # 记录同一次深搜时，当前顶点是否被访问过
        order = []                          # 用于存储拓扑序列
        has_cycle = False                   # 用于判断是否存在环
        
        def dfs(u):
            nonlocal has_cycle
            if u in on_stack:               # 同一次深度优先搜索时，当前顶点被访问过，说明存在环
                has_cycle = True
            if u in visited or has_cycle:   # 当前节点被访问或者有环时直接返回
                return
            
            visited.add(u)                  # 标记节点被访问
            on_stack.add(u)                 # 标记本次深搜时，当前顶点被访问
    
            for v in graph[u]:              # 遍历顶点 u 的邻接顶点 v
                dfs(v)                      # 递归访问节点 v
                    
            order.append(u)                 # 后序遍历顺序访问节点 u
            on_stack.remove(u)              # 取消本次深搜时的 顶点访问标记
        
        for u in graph:
            if u not in visited:
                dfs(u)                      # 递归遍历未访问节点 u
        
        if has_cycle:                       # 判断是否存在环
            return []                       # 存在环，无法构成拓扑序列
        order.reverse()                     # 将后序遍历转为拓扑排序顺序
        return order                        # 返回拓扑序列


    def find_order(self, n:int, edges):
        """ 构建图
        """
        graph = dict()
        for i in range(n):
            graph[i] = []

        for u, v in edges:
            graph[u].append(v)

        return graph


