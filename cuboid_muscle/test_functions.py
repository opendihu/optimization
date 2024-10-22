import matplotlib.pyplot as plt
import numpy as np

def f1(x):
    return np.sin(5*x)**2

def f2(x):
    return x + 0.5*x**2 * np.sin(18*x)

def f3(x):
    return 1-np.abs(x-0.5)

def f4(x):
    return np.sin(np.arctan(x)**0.5)/(1+x**2)

def f5(x):
    return np.sqrt(x)-np.exp(5*(x-1))

x = np.linspace(0,1,1000)
y1 = f1(x)
y2 = f2(x)
y3 = f3(x)
y4 = f4(x)
y5 = f5(x)

print("f6 maximizer: ", x[np.argmax(y1)], " and maximum: ", np.max(y1))
print("f7 maximizer: ", x[np.argmax(y2)], " and maximum: ", np.max(y2))
print("f8 maximizer: ", x[np.argmax(y3)], " and maximum: ", np.max(y3))
print("f9 maximizer: ", x[np.argmax(y5)], " and maximum: ", np.max(y5))

plt.plot(x,y1, label=r"$\hat{f}_6$")
plt.plot(x,y2, label=r"$\hat{f}_7$")
plt.plot(x,y3, label=r"$\hat{f}_8$")
#plt.plot(x,y4, label=r"$\hat{f}_4$")
plt.plot(x,y5, label=r"$\hat{f}_9$")
plt.legend()
plt.show()