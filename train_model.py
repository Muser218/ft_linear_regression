import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
data = pd.read_csv('../data.csv')

# 2. 데이터 전처리
X = data['km'].values
y = data['price'].values

# 3. 모델 파라미터 초기화
learning_rate = 0.001
iterations = 10000
m = len(y)
# theta = np.zeros(2)
theta = np.array([1.0, 1.0])

# 데이터 정규화
X_mean = np.mean(X)
X_std = np.std(X)
X = (X - X_mean) / X_std

# 4. 경사 하강법 알고리즘 구현
for _ in range(iterations):
    # 예측값 계산
    y_pred = theta[0] + theta[1] * X

    # 오차 계산
    error = y_pred - y

    # 파라미터 업데이트
    gradient_0 = (1/m) * np.sum(error)
    gradient_1 = (1/m) * np.sum(error * X)
    theta[0] -= learning_rate * gradient_0
    theta[1] -= learning_rate * gradient_1

    # 디버깅용 코드
    # if _ % 100 == 0:
    #    print(f'gradient_0 : {gradient_0}')
    #    print(f'gradient_1: {gradient_1}')
    #    print(f'epoch: {_}, theta0: {theta[0]}, theta1: {theta[1]}')

# 정규화된 theta0와 theta1
normalized_theta0 = theta[0]
normalized_theta1 = theta[1]

# 역정규화를 통해 원래 스케일로 복원
theta0_restored = normalized_theta0 - normalized_theta1 * X_mean / X_std
theta1_restored = normalized_theta1 / X_std

# 복원된 theta0와 theta1 출력
print(f"theta0: {theta0_restored}")
print(f"theta1: {theta1_restored}")
with open('theta.csv', 'w') as file:
    file.write(f"{theta0_restored},{theta1_restored}\n")
print("theta0와 theta1의 값을 파일에 저장.")

# 5. 결과 시각화
# plt.scatter(X*X_std + X_mean, y, label='Real Price')
# plt.plot(X*X_std + X_mean, (theta[0] + theta[1] * X),
#         color='red', label='Predict Price')
# plt.xlabel('(km)')
# plt.ylabel('Price')
# plt.legend()
# plt.title('Car Price Predict')
# plt.show()
