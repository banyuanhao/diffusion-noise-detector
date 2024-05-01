import numpy as np
from scipy.spatial.distance import cdist

def energy_distance(x, y):
    """计算两个数据集的能量距离"""
    x, y = np.asarray(x), np.asarray(y)
    xy_distance = np.mean(cdist(x, y, 'euclidean'))
    xx_distance = np.mean(cdist(x, x, 'euclidean'))
    yy_distance = np.mean(cdist(y, y, 'euclidean'))
    return 2 * xy_distance - xx_distance - yy_distance

def permutation_test(x, y, num_permutations=1000):
    """执行排列检验来计算能量距离的p值"""
    combined = np.vstack([x, y])
    original_distance = energy_distance(x, y)
    count = 0
    
    for _ in range(num_permutations):
        np.random.shuffle(combined)  # 随机打乱数据
        new_x = combined[:len(x)]
        new_y = combined[len(x):]
        permuted_distance = energy_distance(new_x, new_y)
        if permuted_distance >= original_distance:
            count += 1
    
    p_value = count / num_permutations
    return original_distance, p_value

# 示例数据
np.random.seed(10)
x = np.random.normal(0.01, 1, (10000, 3))
y = np.random.normal(0, 1, (10000, 3))

# 计算能量距离和p值
distance, p_value = permutation_test(x, y, 1000)
print(f"Energy Distance: {distance}")
print(f"P-value: {p_value}")