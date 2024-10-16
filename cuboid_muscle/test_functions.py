import matplotlib.pyplot as plt
import numpy as np

def f1(x):
    return -3*x*(x-1.3) + 0.3

def f2(x):
    return np.exp(-(5*x-3)**2) + 0.2*np.exp(-(30*x-22)**2)

def f3(x):
    return np.exp(-(5*x-5)**2) * np.sin(5*x-1.5) +x

def f4(x):
    return np.exp( -(10*x -2)**2 ) + np.exp(-(10*x-6)**2/10) + 1/((10*x)**2 +1)

def f5(x):
    return 0.5-3*x*(x-1)*np.sin(5*x)

x = np.linspace(0,1,1000)
y1 = f1(x)
y2 = f2(x)
y3 = f3(x)
y4 = f4(x)
y5 = f5(x)

plt.plot(x,y1, label=r"$\hat{f}_1$")
plt.plot(x,y2, label=r"$\hat{f}_2$")
plt.plot(x,y3, label=r"$\hat{f}_3$")
plt.plot(x,y4, label=r"$\hat{f}_4$")
plt.plot(x,y5, label=r"$\hat{f}_5$")
plt.legend()
plt.show()