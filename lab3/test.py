import numpy as np

x=np.array([[1,2],[1,2]])
y=np.array([100,100])
print(np.array([y]))
f=np.concatenate((x, y.T), axis=1)
print(f)

print(f[:,:-1])