import numpy as np

th1 = 0
th2 = 0
th3 = -90

th1 = np.deg2rad(th1)
th2 = np.deg2rad(th2)
th3 = np.deg2rad(th3)

c1 = np.cos(th1)
s1 = np.sin(th1)
c2 = np.cos(th2)
s2 = np.sin(th2)
c3 = np.cos(th3)
s3 = np.sin(th3)

d2 = 46
a2 = 118

t01 = np.array([[c1, -s1, 0, 0], [s1, c1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

t12 = np.array([[c2, -s2, 0, 0], [0, 0, 1, d2], [-s2, -c2, 0, 0], [0, 0, 0, 1]])

t23 = np.array([[c3, -s3, 0, a2], [s3, c3, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

t3e = np.array([[1, 0, 0, a2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

#t1e = t01*t12

t02 = np.dot(t01,t12)
t03 = np.dot(t02,t23)
t0e = np.dot(t03,t3e)

#np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

print("End position: \n")
print(t0e)
