import json
import numpy as np

# 计算user1和user2的欧式距离分数
def euclidean_score(dataset, user1, user2):
    # 用户是否在数据库中出现
    if user1 not in dataset:
        raise TypeError('User ' + user1 + ' not present in the dataset')

    if user2 not in dataset:
        raise TypeError('User ' + user2 + ' not present in the dataset')

    # 提取两个用户评分过的电影
    rated_by_both = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            rated_by_both[item] = 1

    num_ratings = len(rated_by_both) 

    # 如果没有两个用户共同评分过的电影，则说明这两个用户之间没有相似度
    if num_ratings == 0:
        return 0

    # 对于每个共同评分，只计算平方和的平方根，并将该值归一化，使得评分值在[0,1]范围内
    squared_difference = []
    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_difference.append(np.square(dataset[user1][item]-dataset[user2][item]))
    
    return 1/(1+np.sqrt(np.sum(squared_difference)))


# 计算皮尔逊相关系数
def pearson_score(dataset, user1, user2):
    # 用户是否在数据库中出现
    if user1 not in dataset:
        raise TypeError('User ' + user1 + ' not present in the dataset')

    if user2 not in dataset:
        raise TypeError('User ' + user2 + ' not present in the dataset')

    # 提取两个用户评分过的电影
    rated_by_both = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            rated_by_both[item] = 1

    num_ratings = len(rated_by_both) 

    # 如果没有两个用户共同评分过的电影，则说明这两个用户之间没有相似度
    if num_ratings == 0:
        return 0

    # 计算相同评分电影的值之和
    user1_sum = np.sum([dataset[user1][item] for item in rated_by_both])
    user2_sum = np.sum([dataset[user2][item] for item in rated_by_both])

    # 计算相同评分电影的平方值之和
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in rated_by_both])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in rated_by_both])

    # 计算数据集乘积之和
    product_sum = np.sum([dataset[user1][item] * dataset[user2][item] for item in rated_by_both])

    # 计算皮尔逊相关系数需要的各种元素
    Sxy = product_sum - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings
    
    if Sxx * Syy == 0:
        return 0

    return Sxy / np.sqrt(Sxx * Syy)