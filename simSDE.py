import numpy as np 
import matplotlib.pyplot as plt
import model 

test = model.OpinionFormation(N = 175, T = 180, nu = 0.78 , alpha0 = 0.01, alpha1 = 1.19, deltax = 0.02, deltat = 0.0625) 




""" 
tBegin=0
tEnd=test.T
dt=test.dt

t = np.arange(tBegin, tEnd, dt)
N = t.size
IC=-0.5

sqrtdt = np.sqrt(dt)
y = np.zeros(N)
y[0] = IC
for i in range(1,N):
    y[i] = y[i-1] + (test.drift(y[i-1])) * dt + (np.sqrt(1/(test.N)*test.diffusion(y[i-1])))*np.random.normal(loc=0.0,scale=sqrtdt)

fig, ax = plt.subplots()
ax.plot(t,y)
ax.set(xlabel='t', ylabel='y',
       title='Euler-Maruyama-Verfahren zur Berechnung eines \n Ornstein-Uhlenbeck-Prozesses mit $\\theta=1$, $\mu=1.2$, $\sigma=0.3$')
ax.grid()
plt.show()

"""