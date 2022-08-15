# -*- coding: utf-8 -*-
# 相似度计算考

from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import pearsonr
import pandas as pd
import numpy as np


def loadData():
    users_item_score = {
        "Alice": {"A": 5, "B": 3, "C": 4, "D": 4},
        "user1": {"A": 3, "B": 1, "C": 2, "D": 3, "E": 3},
        "user2": {"A": 4, "B": 3, "C": 4, "D": 3, "E": 5},
        "user3": {"A": 3, "B": 3, "C": 1, "D": 5, "E": 4},
        "user4": {"A": 1, "B": 5, "C": 5, "D": 2, "E": 1},
        # 'user5': {'A': 5, 'B': 3, 'C': 4, 'D': 4, 'F': 2},
    }
    return users_item_score


user_data = loadData()
similarity_matrix = pd.DataFrame(
    np.identity(len(user_data)), index=user_data.keys(), columns=user_data.keys(),
)

# UserCF
for u1, items1 in user_data.items():
    for u2, items2 in user_data.items():
        if list(user_data.keys()).index(u1) >= list(user_data.keys()).index(u2):
            continue
        vec1, vec2 = [], []
        for item, rating1 in items1.items():
            rating2 = items2.get(item, -1)
            if rating2 == -1:
                continue
            vec1.append(rating1)
            vec2.append(rating2)
        similarity_matrix[u1][u2] = np.corrcoef(vec1, vec2)[0][1]
        # similarity_matrix[u1][u2] = cosine_distances([vec1, vec2])[0][1]

target_user = "Alice"
num = 2
target_sim_user = (
    similarity_matrix[target_user]
    .sort_values(ascending=False)[1 : num + 1]
    .index.to_list()
)

target_item = "E"
target_user_avg_score = np.mean(list(user_data[target_user].values()))
target_sim_user_sim = []
target_sim_user_div = []
for user in target_sim_user:
    target_sim_user_sim.append(similarity_matrix[target_user][user])
    target_sim_user_div.append(
        user_data[user][target_item] - np.mean(list(user_data[user].values()))
    )
target_item_predScore = target_user_avg_score + np.dot(
    target_sim_user_sim, target_sim_user_div
) / np.sum(target_sim_user_sim)

print(target_item_predScore)
print(user_data)
print(similarity_matrix)
print(target_sim_user)
print("-%" * 40)

# itemCF
item_data = pd.DataFrame(user_data).T.to_dict()
item_data[target_item].pop(target_user)

similarity_matrix = pd.DataFrame(
    np.identity(len(item_data)), index=item_data.keys(), columns=item_data.keys(),
)

for item1, users1 in item_data.items():
    for item2, users2 in item_data.items():
        if list(item_data.keys()).index(item1) >= list(item_data.keys()).index(item2):
            continue
        vec1, vec2 = [], []
        for user in users1.keys():
            if user in users2.keys():
                vec1.append(users1.get(user))
                vec2.append(users2.get(user))
                similarity_matrix[item1][item2] = np.corrcoef(vec1, vec2)[0][1]

target_sim_item = list(
    similarity_matrix.loc[target_item, :]
    .sort_values(ascending=False)[1 : num + 1]
    .index
)
target_item_avg_score = np.mean(list(item_data[target_item].values()))
target_sim_item_sim = []
target_sim_item_div = []
for item in target_sim_item:
    target_sim_item_sim.append(similarity_matrix[item][target_item])
    target_sim_item_div.append(
        item_data[item][target_user] - np.mean(list(item_data[item].values()))
    ) 
target_user_predScore = target_item_avg_score + np.dot(
    target_sim_item_sim, target_sim_item_div
) / np.sum(target_sim_item_sim)

print(target_user_predScore)

