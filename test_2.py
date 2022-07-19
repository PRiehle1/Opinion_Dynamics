import matplotlib.pyplot as plt 
import numpy as np 
nu = 0.8
alpha_0 = np.arange(-0.05,0.05, 0.01)
alpha_1 = np.arange(0.1, 2, 0.1)

results = []

for a0 in range(len(alpha_0)):
    for a1 in range(len(alpha_0)):
        results.append(nu*np.exp(alpha_0[a0] + alpha_1[a1]) - nu*np.exp(-alpha_0[a0] + -alpha_1[a1]) )

plt.plot(results)
plt.show()