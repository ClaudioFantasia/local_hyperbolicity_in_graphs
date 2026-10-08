import numpy as np 
from scipy.special import logsumexp, softmax
import matplotlib.pyplot as plt
beta = 1

delta = np.array(100 * [0] + 50 * [1] + 30 * [2] + 20 * [3])
delta_sampled = np.array(50 * [0] + 25 * [1] + 15 * [2] + 10 * [3])


mu = softmax(delta / beta)
sum = np.sum(mu * delta)
print(f"The sum is {sum}")

mu_sampled = softmax(delta_sampled / beta)
sum_sampled = np.sum(mu_sampled * delta_sampled)
print(f"The sampled sum is {sum_sampled}")


fig = plt.figure(figsize=(6,6))
plt.hist(mu, bins=50)
plt.show()

print(set(mu))
print(set(mu_sampled))

fig = plt.figure(figsize=(6,6))
plt.hist(mu_sampled, bins=50)
plt.show()
