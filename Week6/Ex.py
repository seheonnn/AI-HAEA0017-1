# 비지도 학습
# ===== 1. dataset loading =====
# win dataset을 사용
import random

from sklearn.datasets import load_wine

wine_data = load_wine()
# wine_data 는 data(feature: X) 와 target(label: y) 으로 구성

X = wine_data.data
y = wine_data.target

print(X.shape)
print(y.shape)
# ===== 2. kmeans clustering with numpy =====

# ===========================================
# pick K random points as centroids
# do {
#     cluster the input points to the nearest centroids
#     compute new centroids from the clusters
# } while (cluster is not stable)
# ===========================================

import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

# kmeans () parameter: 데이터셋 (X), cluster 수 (k), max_iters
# return 할 값: clustering 의 결과 --> labels
def kmeans (X, k, max_iters = 100): # k-means clustering 은 cluster 의 개수를 조정하는 작업 밖에 하지 못함
    # data 의 갯수: m <- X.shape[0]
    # 특징의 수: n <- X.shape[1]
    m, n = X.shape

    # pick k random points as centroids -> X 중에서 random 하게 k 개를 선택
    centroids_id = np.random.choice(m, k, replace=False) # m 개 중에서 k 개를 뽑고 한 번 뽑힌 수는 대체 불가 / k=3이므로 3개짜리 벡터임
    centroids = X[centroids_id] # 3 x 13 배열

    for i in range(max_iters):
        # cluster the input points (X) to the nearest centroids
        # 모든 X 에 대해서 각 centroids 까지의 거리를 계산
        # 이 거리들 중에서 가장 최솟값을 가진 centroids 를 X 의 label 로 결정
        distances = [] # 모든 X 에서 모든 centroid 까지의 거리 저장: m x k 배열
        for s in range(m):
            distance = [] # s 번째 X 에서 모든 centroid 까지의 거리
            for j in range(k):
                distance.append(euclidean_distance(X[s], centroids[j])) # s 와 centroids[i] 의 거리
            distances.append(distance)
        labels = np.argmin(distances, axis=1) # 가장 가까운 distance 를 주는 centroid 의 index

        # compute new centroids from the clusters
        # 같은 label 을 갖는 centroid 를 다시 계산
        new_centroids = np.array([X[labels == s].mean(axis=0) for s in range(k)])

        # stable? centroids = new centroids
        if (np.array_equal(centroids, new_centroids)):
            print("break in {}".format(i))
            break

        centroids = new_centroids

    return labels
num_clusters = 3
labels = kmeans(X, num_clusters, max_iters=50)

unique_labels, label_count = np.unique(labels, return_counts=True)
# 초기값을 랜덤하게 고르기 때문에 결과가 매번 달라짐
print(len(unique_labels))
print(label_count)

# 2.1 결과 비교
# groundtruth 의 label 수 계산
cnt0 = cnt1 = cnt2 = 0
for i in range(len(X)):
    cnt0 += (y[i] == 0)
    cnt1 += (y[i] == 1)
    cnt2 += (y[i] == 2)
print(cnt0, cnt1, cnt2)

# labes 의 label 수 계산
cnt0 = cnt1 = cnt2 = 0
for i in labels:
    cnt0 += (i == 0)
    cnt1 += (i == 1)
    cnt2 += (i == 2)
print(cnt0, cnt1, cnt2)

# ===== 3. kmeans clustering with sklearn =====
from sklearn.cluster import KMeans

kmeansk = KMeans(n_clusters=num_clusters)
kmeansk.fit(X) # 결과는 kmeansk.labels_ 에 저장됨

unique_labels, label_count = np.unique(kmeansk.labels_, return_counts=True)
print(len(unique_labels))
print(label_count)

# ===== 4. dbscan with numpy =====

# ===========================================
# C =0
# for each point in the set
#     p.label = -1 // Noise
# for each noise point p in the set
#     if (p is not a core point)
#         continue
#     C = C + 1
#     p.label = C
#     seed set S <- p ∪ p.nbhd
#     for each point q in S
#         if (q.label == -1)
#             q.label = C
#         S <- S ∪ q.nbhd
# ===========================================

# X 의 점들 중에서 point 와의 거리가 eps 이내인 점들을 찾아서 리턴
def get_neighbors(X, point, eps):
    neighbors = []
    for i, candidate in enumerate(X):
        if(euclidean_distance(point, candidate) <= eps):
            neighbors.append(i)
    return neighbors

def trace_border(X, labels, point, neighbors, n_clusters, eps, min_points):
    i = 0
    while i < len(neighbors):
        neighbor_point = neighbors[i]
        if labels[neighbor_point] == -1:
            labels[neighbor_point] = n_clusters
            new_neighbors = get_neighbors(X, X[neighbor_point], eps)
            if len(new_neighbors) >= min_points:
                new_neighbors += neighbors
        i += 1

# dbscan 은 kmeans 와 달리 cluster 의 수가 미정
# dbscan 은 cluster 의 수와 label 을 return
def dbscan(X, eps, min_points):
    m, n = X.shape
    labels = np.full(m, -1)
    n_clusters = 0

    for i, point in enumerate(X):
        if(labels[i] >= 0):
            continue
        # core point 인지 아닌지 판단 -> point 의 neighbor 를 계산해서, 그 갯수가 num_points 보다 크면 core point
        # neighbor -> point 와의 거리가 eps 이내인 점
        neighbors = get_neighbors(X, point, eps)
        if(len(neighbors) < min_points): # Not core point -> continue
            continue

        n_clusters += 1
        labels[i] = n_clusters

        trace_border(X, labels, point, neighbors, n_clusters, eps, min_points)

    return n_clusters, labels

eps = 30
min_points = 4
n_labels, labels = dbscan(X, eps, min_points)

unique_labels, label_count = np.unique(labels, return_counts=True)
# print(n_labels)
print(len(unique_labels))
print(label_count)

# ===== 5. dbscan with sklearn =====
from sklearn.cluster import DBSCAN

dbscansk = DBSCAN(eps=30, min_samples=4)
dbscansk.fit(X)
unique_labels, label_count = np.unique(dbscansk.labels_, return_counts=True)
print(len(unique_labels))
print(label_count)