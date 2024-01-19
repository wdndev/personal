""" 背包问题
"""

class ZeroOnePack:
    """ 0-1背包问题
    """
    def zero_ont_pack_method1(self, weight:[int], value:[int], W:int):
        """ 思路1 : 动态规划 + 二维基本思路
        """
        size = len(weight)
        dp = [[0 for _ in range(W + 1)] for _ in range(size + 1)]

        # 枚举前i中物品
        for i in range(1, size + 1):
            # 枚举背包装载重量
            for w in range(W + 1):
                # 第 i - 1件物品装不下
                if w < weight[i - 1]:
                    # dp[i][w] 取「前 i - 1 件物品装入载重为 w 的背包中的最大价值」
                    dp[i][w] = dp[i - 1][w]
                else:
                    # dp[i][w] 取「前 i - 1 件物品装入载重为 w 的背包中的最大价值」
                    # 与「前 i - 1 件物品装入载重为 w - weight[i - 1] 的背包中，
                    # 再装入第 i - 1 物品所得的最大价值」两者中的最大值
                    dp[i][w] = max(dp[i-1][w], dp[i-1][w - weight[i-1]] + value[i-1])

        return dp[size][W]
    
    def zero_ont_pack_method2(self, weight:[int], value:[int], W:int):
        """ 思路2 : 动态规划 + 滚动数组优化
        """
        size = len(weight)
        dp = [0 for _ in range(W + 1)]

        # 枚举前i种物品
        for i in range(1, size+1):
            # 逆序枚举背包装载重量（避免错误值状态）
            for w in range(W, weight[i - 1] - 1, -1):
                # dp[w] 取「前 i - 1 件物品装入载重为 w 的背包中的最大价值」
                # 与「前 i - 1 件物品装入载重为 w - weight[i - 1] 的背包中，
                # 再装入第 i - 1 物品所得的最大价值」两者中的最大值
                dp[w] = max(dp[w], dp[w-weight[i-1]] + value[i-1])

        return dp[W]

class CompletePack:
    """ 完全背包问题
    """
    def complete_pack_method1(self, weight:[int], value:[int], W:int):
        """ 思路1：动态规划 + 二维基本思路
        """
        size = len(weight)
        dp = [[0 for _ in range(W + 1)] for _ in range(size + 1)]

        # 枚举前i种物品
        for i in range(1, size + 1):
            # 枚举背包装载重量
            for w in range(W + 1):
                # 枚举第 i-1种物品能取个数
                for k in range(w // weight[i - 1] + 1):
                    # dp[i][w] 取所有 dp[i - 1][w - k * weight[i - 1] + k * value[i - 1] 中最大值
                    dp[i][w] = max(dp[i][w], dp[i-1][w-k*weight[i-1]] + k*value[i-1])
        
        return dp[size][W]
    
    def complete_pack_method2(self, weight:[int], value:[int], W:int):
        """ 思路2：动态规划 + 状态转移方程优化
        """
        size = len(weight)
        dp = [[0 for _ in range(W+1)] for _ in range(size + 1)]

        # 枚举前i种物品
        for i in range(1, size + 1):
            # 枚举背包状态重量
            for w in range(W + 1):
                # 第 i - 1 件物品装不下
                if w < weight[i - 1]:
                    # dp[i][w]取“前i-1种物品装入载重为w的背包种的最大价值”
                    dp[i][w] = dp[i - 1][w]
                else:
                    # dp[i][w] 取「前 i - 1 种物品装入载重为 w 的背包中的最大价值」
                    # 与「前 i 种物品装入载重为 w - weight[i - 1] 的背包中，
                    # 再装入 1 件第 i - 1 种物品所得的最大价值」两者中的最大值
                    dp[i][w] = max(dp[i-1][w], dp[i][w-weight[i-1]] + value[i-1])
        
        return dp[size[W]]
    
    def completePackMethod3(self, weight: [int], value: [int], W: int):
        """ 思路 3：动态规划 + 滚动数组优化
        """
        size = len(weight)
        dp = [0 for _ in range(W + 1)]
        
        # 枚举前 i 种物品
        for i in range(1, size + 1):
            # 正序枚举背包装载重量
            for w in range(weight[i - 1], W + 1):
                # dp[w] 取「前 i - 1 种物品装入载重为 w 的背包中的最大价值」
                # 与「前 i 种物品装入载重为 w - weight[i - 1] 的背包中，
                # 再装入 1 件第 i - 1 种物品所得的最大价值」两者中的最大值
                dp[w] = max(dp[w], dp[w - weight[i - 1]] + value[i - 1])
                
        return dp[W]

class MultiplePack:
    """ 多重背包问题
    """
    def multiple_pack_method1(self, weight:[int], value:[int], count:[int], W:int):
        """ 思路1：动态规划 + 二维基本思路
        """
        size = len(weight)
        dp = [[0 for _ in range(W + 1)] for _ in range(size + 1)]

        # 枚举前i种物品
        for i in range(1, size+1):
            # 枚举背包装载重量
            for w in range(W + 1):
                # 枚举第i-1种物品能取的个数
                for k in range(min(count[i - 1], w // weight[i-1] - 1)):
                    # dp[i][w]取所有dp[i-1][w-k*weight[i-1]] + k* value[i-1]中最大值
                    dp[i][w] = max(dp[i][w], dp[i-1][w-k*weight[i-1]] + k*value[i-1])

        return dp[size][W]
    
    def multiplePackMethod2(self, weight: [int], value: [int], count: [int], W: int):
        """ 思路 2：动态规划 + 滚动数组优化
        """
        size = len(weight)
        dp = [0 for _ in range(W + 1)]
        
        # 枚举前 i 种物品
        for i in range(1, size + 1):
            # 逆序枚举背包装载重量（避免状态值错误）
            for w in range(W, weight[i - 1] - 1, -1):
                # 枚举第 i - 1 种物品能取个数
                for k in range(min(count[i - 1], w // weight[i - 1]) + 1):
                    # dp[w] 取所有 dp[w - k * weight[i - 1]] + k * value[i - 1] 中最大值
                    dp[w] = max(dp[w], dp[w - k * weight[i - 1]] + k * value[i - 1])
                
        return dp[W]
        



