# ===== 1. import library =====
import numpy as np
import pandas as pd
# from google.colab import drive

# ===== 2. upload file (google colab 사용시) =====
# from google.colab import files
# files.upload()

# ===== 3. read csv file =====
nbastat = pd.read_csv('./nbastat2022.csv')
print(type(nbastat))

# ===== 4. 줄 수 세기 -> m =====
m = len(nbastat)
print(m)

# ===== 5. feature 선택 -> FGA, 3PA, FTA, PTS =====
# x1: FGA 슛 시도
# x2: 3PA 3점 슛 시도
# x3: FTA
# y: PTS 몇 득점 했는지
x1 = nbastat[['FGA']]
x2 = nbastat[['3PA']]
x3 = nbastat[['FTA']]
y = nbastat[['PTS']]

print(type(y))

# ===== 6. 결측값 제거 =====
x1 = x1.dropna(axis=0) # 결측값을 포함한 행 삭제
x2 = x2.dropna(axis=0)
x3 = x3.dropna(axis=0)
y = y.dropna(axis=0)

print(len(x1))

m = len(x1)

# ===== 7. pandas의 dataframe -> np의 array로 변환 =====
x1 = (np.array(x1)).reshape(m, 1)
x2 = (np.array(x2)).reshape(m, 1)
x3 = (np.array(x3)).reshape(m, 1)
y = (np.array(y)).reshape(m, 1)

print(type(x1))

# ===== 8. 변수 설정: Xb = (1 x1 x2 x3) =====
# 열이 1인 vector 만들기
x0 = np.ones((m,1))

Xb = np.c_[x0, x1, x2, x3] # m, 1 vector 4 개로 m, 4 짜리 vector(tensor)를 만듦

print(x1.shape)
print(Xb.shape)

# ===== 9. hyperparameter 준비 =====
# learning_rate 나 n_iter 를 늘림으로써 loss, error를 줄일 수 있음
learning_rate = 0.00001

n_iter = 200

# ===== 10. parameter 초기화 =====
theta = np.zeros((4, 1))
gradients = np.zeros((4, 1))

print(theta.shape)

# ===== 11. J (loss 함수) 계산 =====
def compute_J (theta, Xb, y):
    # loss vector: 예측값 - 실측값
    # 예측값: theta * Xb -> Xb.dot(theta)
    # 실측값: y
    loss_vector = Xb.dot(theta) - y
    loss_vector = loss_vector.reshape(-1) # Xb와 y의 차원을 맞춰야
    # loss vector 의 제곱
    loss2 = np.square(loss_vector)
    # 더하고 (1/2m) 곱하기
    loss_sum = np.sum(loss2) / (2*m)
    return loss_sum

# ===== 12. 훈련 =====
loss_arr = []
for i in range(n_iter):
    # gradient 계산
    gradients = (1.0/m)*Xb.T.dot(Xb.dot(theta) - y) # 단변량 선형 회귀의 gradients 와 동일
    # gradient descent method 적용
    theta = theta - learning_rate * gradients
    # loss 계산
    loss = compute_J(theta, Xb, y)
    # loss 값 저장
    loss_arr.append(loss)

print(loss_arr)

# ===== 13. 결과 출력 =====
import matplotlib.pyplot as plt
plt.plot(loss_arr, '.b')
plt.show()

# error 출력: error = (예측값 - 실측값)^2 의 평균
y_hat = Xb.dot(theta)

error = (1/ len(y_hat)) * (y_hat - y).T.dot(y_hat - y)

print(error)