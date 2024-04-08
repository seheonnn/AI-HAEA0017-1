# ===== 1. import libraries =====
import numpy as np
import pandas as pd

# ===== 2. data preparation -> (1) 데이터 불러 오기 =====ㄴ
nbastat = pd.read_csv('./nba_stats_2020_to_2023.csv')
m = len(nbastat)
print(m)

# ===== 2. data preparation -> (2) 특징 선택 =====
selected_columns = ['Pos', 'FG%', '3P%', 'TRB', 'AST', 'STL', 'BLK']

nbastat2 = nbastat[selected_columns]

print(nbastat2)

# ===== 2. data preparation -> (3) 결측치 제거 =====
# isna() 함수
rows_with_na = nbastat2[nbastat2.isna().any(axis=1)]

print(rows_with_na)

print(len(nbastat2))

# dropna() 함수
nbastat2 = nbastat2.dropna(axis=0)
print(len(nbastat2))

# ===== 2. data preparation -> (4) 이상치 제거 =====
# Pos이 C, PG, SG, SF, PF가 아닌 행을 제거
# Pos이 C, PG, SG, SF, PF인 행만 선택

print(len(nbastat2))

nbastat3 = nbastat2[(nbastat2['Pos'] == 'C') | (nbastat2['Pos'] == 'PG') | (nbastat2['Pos'] == 'SG') | (nbastat2['Pos'] == 'SF') | (nbastat2['Pos'] == 'PF')]

print(len(nbastat3))

# ===== 2. data preparation -> (5) 변수 설정 =====
# FG% 3P% TRB AST STL BLK -> feature 들
x1 = nbastat3[['FG%']]
x2 = nbastat3[['3P%']]
x3 = nbastat3[['TRB']]
x4 = nbastat3[['AST']]
x5 = nbastat3[['STL']]
x6 = nbastat3[['BLK']]
y = nbastat3[['Pos']]

m = len(y)
print(m)

# ===== 이상치 제거 확인 =====
# y의 값이 C, PG, SG ,PF, SF만 있는지 확인
unique = y.drop_duplicates()
print(unique)

# binary classification :  C를 class 1, 그 이외의 값을 class 0으로 labeling (Center 인지 아닌지)
# multi-class classification : C는 2, PF나 SF는 1, PG나 SG는 0으로 labeling
# yb : binary classification용 변수 : C는 1, 그외는 0
# yt : multiclass classification용 변수 : C는 2, PF, SF는 1, PG, SG는 0

yb = nbastat3['Pos'].apply(lambda x : 1 if x == 'C' else 0)

yt = nbastat3['Pos'].map({'C' : 2, 'PF' : 1, 'SF' : 1, 'PG' : 0, 'SG' : 0})

# ===== 2. data preparation -> (6) 변수 벡터 설정 =====
x0 = np.ones((m, 1))
Xb = np.c_[x0, x1, x2, x3, x4, x5, x6]

yb = (np.array(yb)).reshape(m, 1)

print(Xb.shape)
print(yb.shape)
print(type(Xb))
print(type(yb))

# ===== 2. data preparation -> (7) train-test set 분할 =====
# np의 random permutation 사용
def permutation_split(X, Y, ratio = 0.7, random_state = 1004):
    # train set 의 크기 : num_a, test set 의 크기 : num_b
    num_a = int(X.shape[0] * ratio)
    num_b = X.shape[0] - num_a

    np.random.seed(random_state)
    shuffle = np.random.permutation(X.shape[0])

    X = X[shuffle,:]
    Y = Y[shuffle,:]

    Xa = X[:num_a]
    Ya = Y[:num_a]
    Xb = X[num_a:]
    Yb = Y[num_a:]

    return Xa, Xb, Ya, Yb

Xb_train, Xb_test, Y_train, Y_test = permutation_split(Xb, yb, 0.6)
print(len(Xb_train))
print(len(Xb_test))
print(len(Y_train))
print(len(Y_test))

# ===== 3. numpy를 이용한 구현 =====
# 3.1 loss 함수 정의
# cross entropy 함수 : -mean ( y * log (y_hat) + (1-y) * log(1 - y_hat) )
# 조심 또 조심 : NaN -> divide-by-zero, -inf, +inf
# log (0) = -inf -> 피해야
# y_hat 의 값이 0이면 log(y_hat) = -inf
# y_hat 의 값이 1이면 log(1 - y_hat) = -inf
# y_hat 의 값을 (0 ~ 1) -> (e ~ 1 - e)
# 1/n -> 1/(n + 0.00001)와 비슷
def loss_CE(y_hat, y):
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1-epsilon)
    return -np.mean(y * np.log(y_hat) + (1-y) * np.log(1 - y_hat))

# 3.2 model 정의
class LogisticRegressionNumpy:
    def __init__(self, learning_rate = 0.0001, n_iter = 1000): # hyperparameter 를 입력
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.theta = None # model 의 parameter

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def train(self, X, y):
        # 데이터의 수(m)와 속성의 수(n)를 결정
        m, n = X.shape
        self.theta = np.zeros((n, 1))

        loss_arr = []
        # gradient descent method
        for _ in range(self.n_iter):
            # y_hat 을 계산
            z = X.dot(self.theta)
            y_hat = self.sigmoid(z)

            # loss 계산
            loss = loss_CE(y_hat, y)
            loss_arr.append(loss)

            # gradient 계산
            gradient = (1/m) * X.T.dot(y_hat - y)

            # gradient descent
            self.theta = self.theta - self.learning_rate * gradient

        return loss_arr

    def predict(self, X):
        # y_hat 을 계산
        z = X.dot(self.theta)
        y_hat = self.sigmoid(z)

        # logistic regression : y_hat 값이 0.5보다 적으면 0, 크면 1을 return
        y_hat_cls = [1 if i > 0.5 else 0 for i in y_hat]
        return y_hat_cls

# 3.3 훈련
model = LogisticRegressionNumpy(learning_rate=0.01,n_iter=100000)
loss_arr = model.train(Xb_train, Y_train)

import matplotlib.pyplot as plt
plt.plot(loss_arr, '.b')
plt.ylabel("loss")
plt.show()

# 3.4 결과 검증
# 정확도 = 정답 / 전체 데이터 수

# train data 에 대한 결과
prediction = model.predict(Xb_train)
cnt = 0
for i in range(len(prediction)):
    cnt += (prediction[i] == Y_train[i])
print(100 * (cnt / len(prediction)))

# test data 에 대한 결과
prediction = model.predict(Xb_test)
cnt = 0
for i in range(len(prediction)):
    cnt += (prediction[i] == Y_test[i])
print(100 * (cnt / len(prediction)))

# ===== 4. pytorch를 이용한 구현 =====
# 4.1 import library
import torch
import torch.nn as nn
import torch.optim as optim

# 4.2 model 정의
class LogisticRegressionPytorch(nn.Module):
    def __init__(self, input_size): # 속성의 개수 (n) -> 7
        super(LogisticRegressionPytorch, self).__init__();
        self.linear = nn.Linear (input_size, 1) # 7 개의 입력 (Xb: 1, .... x6)을 1개의 출력 (z)으로 대응하는 선형 함수: z = theta * Xb
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # 결과를 출력하는 함수 : y_hat = sigmoid(-theta * Xb) -> sigmoid(forward(x))
        y_hat = self.sigmoid(self.linear(x))
        return y_hat

    def predict(self, x): # y_hat -> y_hat의 값이 0.5보다 작으면 0, 크면 1을 리턴
        y_hat = self.forward(x)
        return [1 if p > 0.5 else 0 for p in y_hat]

# 4.3 train 함수 정의
def train(model, X, y, learning_rate = 0.0001, n_iter = 10000):
    # loss 함수와 gradient method를 선언
    criterion = nn.BCELoss() # loss 함수
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # X와 Y의 type을 변경 : X와 y는 numpy의 ndarray -> pytorch의 tensor type으로 변경
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    loss_arr = []
    for _ in range(n_iter):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y) # loss pytorch의 데이터 타입임
        loss_arr.append(loss)

        # gradient 구하기
        loss.backward()

        # gradient descent 수행하기
        optimizer.step()

    return loss_arr

# 4.4 training
model = LogisticRegressionPytorch(input_size=7)
loss_arr = train(model, Xb_train, Y_train, learning_rate=0.01, n_iter=100000)

# 4.5 결과 검증
# 정확도 = 정답 / 전체 데이터 수

# train data 에 대한 결과
X = torch.tensor(Xb_train, dtype=torch.float32)
prediction = model.predict(X)
cnt = 0
for i in range(len(prediction)):
    cnt += (prediction[i] == Y_train[i])
print(100 * (cnt / len(prediction)))

# test data 에 대한 결과
X = torch.tensor(Xb_test, dtype=torch.float32)
prediction = model.predict(X)
cnt = 0
for i in range(len(prediction)):
    cnt += (prediction[i] == Y_test[i])
print(100 * (cnt / len(prediction)))

# ===== 5. scikit-learn을 이용한 구현 =====
# 5.1 import library
from sklearn.linear_model import LogisticRegression

# 5.2 model 정의 및 훈련
model = LogisticRegression(max_iter=1000)
model.fit(Xb_train, Y_train.ravel()) # ravel : 1차원 배열을 LogisticRegression의 형태에 맞춤

# 5.3 결과 검증
# 정확도 = 정답 / 전체 데이터 수

# train data 에 대한 결과
prediction = model.predict(Xb_train)
cnt = 0
for i in range(len(prediction)):
    cnt += (prediction[i] == Y_train[i])
print(100 * (cnt / len(prediction)))

# test data 에 대한 결과
prediction = model.predict(Xb_test)
cnt = 0
for i in range(len(prediction)):
    cnt += (prediction[i] == Y_test[i])
print(100 * (cnt / len(prediction)))