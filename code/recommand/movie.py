import json
import numpy as np
from estimate import pearson_score

# 为给定用户生成电影推荐
def generate_recommendations(dataset, user):
    if user not in dataset:
        raise TypeError('User ' + user + ' not present in the dataset')

    total_scores = {}
    similarity_sums = {}

    for u in [x for x in dataset if x != user]:
        similarity_score = pearson_score(dataset, user, u)

        if similarity_score <= 0:
            continue

        # 找到还未被该用户评分的电影
        for item in [x for x in dataset[u] if x not in dataset[user] or dataset[user][x] == 0]:
            total_scores.update({item: dataset[u][item] * similarity_score})
            similarity_sums.update({item: similarity_score})
    
    # 如果该用户看过数据库中的所有电影，那就不能为用户 推荐电影
    if len(total_scores) == 0:
        return ['No recommendations possible']

    # 生成一个电影评分标准化列表
    movie_ranks = np.array([[total/similarity_sums[item], item] 
            for item, total in total_scores.items()])

    # 根据第一列对皮尔逊相关系数进行降序排列
    movie_ranks = movie_ranks[np.argsort(movie_ranks[:, 0])[::-1]]

    # 提取出推荐电影
    recommendations = [movie for _, movie in movie_ranks]

    return recommendations


data_file = 'movie_ratings.json'
with open(data_file, 'r') as f:
    data = json.loads(f.read())

user = 'Michael Henry'
print("\nRecommendations for " + user + ":")
movies = generate_recommendations(data, user) 
for i, movie in enumerate(movies):
    print(str(i+1) + '. ' + movie)

user = 'John Carson' 
print("\nRecommendations for " + user + ":")
movies = generate_recommendations(data, user) 
for i, movie in enumerate(movies):
    print(str(i+1) + '. ' + movie)