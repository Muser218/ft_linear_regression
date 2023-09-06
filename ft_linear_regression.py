import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_data(filename):
    open_file = open(filename, 'r')
    data = open_file.read()
    data = data.split('\n')
    data = data[1:]
    data = [i.split(',') for i in data]
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
    data = normalize_data(data)
    data = add_intercept(data)
    theta = np.zeros(2)
    theta = gradient_descent(data, theta, 0.1, 1000)
    print(theta)
    plt.scatter(data[:, 1], data[:, 2])
    plt.plot(data[:, 1], predict(data, theta), color='red')
    plt.show()

if __name__ == "__main__":
    main()

