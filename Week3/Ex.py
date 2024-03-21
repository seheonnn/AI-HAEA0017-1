import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

# ===== 파일 내용 확인 =====
nbastat = pd.read_csv('./nbastat2022.csv')
# print(nbastat)

# ===== 줄 수 count =====
m = len(nbastat)
# print(m) # m : 데이터의 수, sample의 수

# ===== feature selection -> nbastat에서 FGA, FGM만 추출 =====
# FGA : 몇 개의 슛을 던지는지
# FGM : 몇 개의 슛을 성공하는지
# nbastat의 많은 column 들 중에서 이 값들만 추출해서 사용
X = nbastat[['FGA']]
Y = nbastat[['FGM']]
print(X)
print(Y)

# ===== 결측값을 처리 =====
# pandas에서 결측값을 해소하는 함수: fillna
X = X.fillna(0)
Y = Y.fillna(0)
print(type(X))
print(type(Y))

# ===== pandas의 dataframe --> numpy의 array로 변환 =====
X = (np.array(X)).reshape(m, 1)
Y = (np.array(Y)).reshape(m, 1)
print(X)
print(type(X))
print(X.shape)

# ===== 그리기 =====
plt.plot(X, Y, '.b') # 소수점 한 자리까지
plt.xlabel("FGA")
plt.ylabel("FGM")
plt.show() # 그려진 그래프의 직선식 -> 선형회귀

# ===== 준비 =====
# 학습률 : learning rate
learning_rate = 0.001

# 훈련(반복) 횟수 : epochs
n_iter = 200

# ===== 초기화 =====
# 인공지능 훈련은 세타의 값을 결정하는 것
# theta와 theta의 미분(gradient)를 초기화 -> 0으로 초기화
theta = np.zeros((2, 1))
gradients = np.zeros((2, 1))

print(theta.shape)

# ===== 변수 설정 =====
# Xb를 설정 -> Xb = (1 X)의 결합
X0 = np.ones((m,1)) # 1로 가득 찬 배열
Xb = np.c_[X0, X] # X0 와 X를 결합

print(Xb.shape)
print(theta.shape)
print(Y.shape)

# ===== 훈련 ppt 54 =====
for i in range(n_iter):
    # 1. Xb * theta -> Xb.dot(theta) -> (249, 2) x (2, 1) -> (249, 1)
    # 2. Xb * theta - Y -> Xb.dot(theta) - Y -> (249, 1) - (249, 1) -> (249, 1)
    # 3. (Xb * theta - Y) * Xb -> (249, 1) * (249, 2) -> 불가
    #     -> Xb^T * (Xb * theta - Y) (2, 249) * (249, 1) -> (2, 1)
    gradients = (1.0/m) * Xb.T.dot(Xb.dot(theta) - Y) # (2, 1)
    theta = theta - learning_rate * gradients

print(theta)

# ===== 결과 가시화 =====
# (X, Y)의 데이터와 Y = theta_0 + X * theta_1
Y_pred = Xb.dot(theta)

plt.plot(X, Y_pred, color = 'Red')
plt.plot(X, Y, '_b')
plt.show()
# 빨간 직선이 예측값 -> 실제 데이터 분포와 유사하도록 훈련 횟수를 조정