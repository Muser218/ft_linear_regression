import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
# import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim

def get_data(filename):
    open_file = open(filename, 'r')
    data = open_file.read()
    data = data.split('\n')
    data = data[1:]
    data = [i.split(',') for i in data]
    data = [i for i in data if len(i) == 2]
    data = [[float(j) for j in i] for i in data]
    data = np.array(data)
    return data

def normalize_data(data):
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return data

def add_intercept(data):
    data = np.insert(data, 0, 1, axis=1)
    return data

def predict(data, theta):
    return np.dot(data, theta)

def cost(data, theta):
    m = len(data)
    y = data[:, 1]
    y_hat = predict(data, theta)
    return np.sum((y_hat - y) ** 2) / (2 * m)

def gradient_descent(data, theta, alpha, n_iterations):
    m = len(data)
    y = data[:, 1]
    for i in range(n_iterations):
        y_hat = predict(data, theta)
        theta = theta - alpha * (1 / m) * np.dot(data.T, (y_hat - y))
    return theta

def main():
    if len(sys.argv) != 2:
        print("usage: python3 test.py [filename]")
        sys.exit()
    try:
        data = get_data(sys.argv[1])
    except:
        print("File not found")
        sys.exit()
    # data = normalize_data(data)
    #data = add_intercept(data)
    #theta = np.zeros(2)
    #theta = gradient_descent(data, theta, 0.1, 1000)
    #print(theta)
    #plt.scatter(data[:, 1], data[:, 2])
    #plt.plot(data[:, 1], predict(data, theta), color='red')
    #plt.show()

    # 데이터
    x_train = torch.FloatTensor([i[0] for i in data])
    y_train = torch.FloatTensor([i[1] for i in data])


    # W_l = np.linspace(-5, 7, 1000)
    # cost_l = []
    # for W in W_l:
    #     hypothesis = W * x_train
    #     cost = torch.mean((hypothesis - y_train) ** 2)

    #     cost_l.append(cost.item())
    # plt.plot(W_l, cost_l)
    # plt.xlabel('$W$')
    # plt.ylabel('Cost')
    # fig = plt.gcf()
    # plt.draw()
    # plt.pause(0.001)
    # input("Press Enter to exit...")






    # 모델 초기화
    W = torch.zeros(1)
    # learning rate 설정
    lr = 0.1

    nb_epochs = 100
    for epoch in range(nb_epochs + 1):
        
        # H(x) 계산
        hypothesis = x_train * W
        
        # cost gradient 계산
        cost = torch.mean((hypothesis - y_train) ** 2)
        gradient = torch.sum((W * x_train - y_train) * x_train)

        # if epoch % 10 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}'.format(
            epoch, nb_epochs, W.item(), cost.item()
        ))

        # cost gradient로 H(x) 개선
        W -= lr * gradient


    # # 모델 초기화
    # W = torch.zeros(1, requires_grad=True)
    # # optimizer 설정
    # # optimizer = optim.SGD([W], lr=0.15)
    # optimizer = optim.SGD([W], lr=0.001)

    # nb_epochs = 100
    # for epoch in range(nb_epochs + 1):
        
    #     # H(x) 계산
    #     hypothesis = x_train * W
        
    #     # cost 계산
    #     cost = torch.mean((hypothesis - y_train) ** 2)

    #     print('Epoch {:4d}/{} W: {:.3f} Cost: {:.6f}'.format(
    #         epoch, nb_epochs, W.item(), cost.item()
    #     ))

    #     # cost로 H(x) 개선
    #     optimizer.zero_grad()
    #     cost.backward()
    #     optimizer.step()
    

    # # Data
    # plt.scatter(x_train, y_train)
    # # Best-fit line
    # xs = np.linspace(1, 3, 1000)
    # plt.plot(xs, xs)

    # fig = plt.gcf()
    # plt.draw()
    # plt.pause(0.001)
    # input("Press Enter to exit...")


if __name__ == "__main__":
    main()
