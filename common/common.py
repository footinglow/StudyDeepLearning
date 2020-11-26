import numpy as np

def cross_entropy_error(y, t):
  delta = 1e-7
  return -np.sum(t*np.log(y+delta))   # np.log(0)が-infとなるのを防ぐため

# 数値微分
def numerical_defferentiation(f, x):
  h = 1e-4     # 微小なhは10のマイナス４乗を用いるとよい結果が得られることが分かっている。
  return (f(x + h) - f(x - h)) / (2 * h)

def mean_squared_error(y, t):
  return 0.5 * np.sum((y - t) ** 2)


def sigmoid(x):
  return 1 / (1 + np.exp( -x ))