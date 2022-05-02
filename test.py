import numpy as np

np.random.seed(2022)
a = [1,2,3,4,5,6,7,8,9,0]
n, k = 3,4
results = np.zeros((n,k))
for row in range(n):
    results[row, :] = np.random.choice(a, size=k, replace=True)
print(results)