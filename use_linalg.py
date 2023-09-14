import sys
import numpy as np
import torch
import statsmodels.api as sm


def get_data(filename):
    open_file = open(filename, 'r')
    data = open_file.read()
    data = data.split('\n')
    data = data[1:]
    data = [i.split(',') for i in data]
    data = [i for i in data if len(i) == 2]
    data = [[float(j) for j in i] for i in data]
    data = torch.FloatTensor(data)
    return data

def main():
    if len(sys.argv) != 2:
        print("usage: python3 test.py [filename]")
        sys.exit()
    try:
        data = get_data(sys.argv[1])
    except:
        print("File not found")
        sys.exit()

    X = [i[0] for i in data]
    X = sm.add_constant(X)
    y = [i[1] for i in data]

    w = np.linalg.inv(X.T @ X) @ X.T @ y
    print(w)


if __name__ == "__main__":
    main()


