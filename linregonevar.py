import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    # Get housing price data
    df = pd.read_csv("house-prices.csv")
    x_train = df["SqFt"]
    y_train = df["Price"]

    # Scaling x_train
    x_train = (x_train / x_train.max())

    y_pred  = np.zeros(np.shape(y_train)[0])
    w,b,cost_list = gradient_descent(x_train,y_train,0.5,10000)
    y_pred = w*x_train + b
    error = cost(y_pred,y_train)

    print(f"Error: {error}")
    print(f"w: {w}, b:{b}")

    # plotting linear regression
    plt.scatter(x_train,y_train)
    plt.plot(x_train,y_pred)
    plt.title("Linear regression model")
    plt.xlabel("Area")
    plt.ylabel("Price")
    plt.show()

    # for seeing how error converge
    plt.plot(cost_list)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost function convergence")
    plt.show()


def cost(y_pred,y):
    m = np.shape(y)[0]
    j = np.sum((y_pred - y)**2)
    return (1/(2*m)) * j


def compute_gradient(x,y,w,b):
    dj_dw = 0
    dj_db = 0
    m = np.shape(x)[0]
    dj_dw = (1/m) * np.sum(((w*x + b) - y) * x)
    dj_db = (1/m) * np.sum((w*x + b) - y)
    return dj_dw,dj_db


def gradient_descent(x,y,lr=0.1,iteration=1000):
    w = 0
    b = 0
    m = np.shape(x)[0]
    cost_list = []
    plt.scatter(x,y)
    for i in range(iteration):
        dj_dw,dj_db = compute_gradient(x,y,w,b)
        w_tmp = (w)-(lr*(1/m))*(dj_dw)
        b_tmp = (b)-(lr*(1/m))*(dj_db)
        w = w_tmp
        b = b_tmp
        y_tmp = w*x + b
        cost_list.append(cost(y_tmp,y))
    return w,b,cost_list


main()