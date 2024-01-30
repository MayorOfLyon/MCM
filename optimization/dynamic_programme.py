def knapsack_01(values, weights, capacity):
    n = len(values)
    
    # 初始化动态规划表
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # 填充动态规划表
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # 如果当前物品的重量大于背包容量，则不能放入
            if weights[i - 1] > w:
                dp[i][w] = dp[i - 1][w]
            else:
                # 在放和不放中选择最优解
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
    
    # 从动态规划表中找到最优解
    optimal_value = dp[n][capacity]
    
    # 找到选中的物品
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(i - 1)
            w -= weights[i - 1]
    
    selected_items.reverse()
    
    return optimal_value, selected_items

# 例子
values = [3,4,5]
weights = [1,2,3]
capacity = 5

optimal_value, selected_items = knapsack_01(values, weights, capacity)

print("最优值:", optimal_value)
print("选中的物品:", selected_items)
