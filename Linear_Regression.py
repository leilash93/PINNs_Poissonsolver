from dolfin import *
import fenics
import pyadjoint
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import Union
import matplotlib.pyplot as plt


# extracting u,x,y arrays for training the neural network
u= fenics_solver(nx=10)
mesh=UnitSquareMesh(10, 10)
W=FunctionSpace(mesh, "CG", 1)
u_array=fenics_to_numpy(u)
u_array1=u_array.reshape(121,1)
x=fenics.interpolate(Expression("x[0]",degree=1),W)
x_array=fenics_to_numpy(x)
x_array1= x_array.reshape(121,1)
y=fenics.interpolate(Expression("x[1]",degree=1),W)
y_array=fenics_to_numpy(y)
y_array1=y_array.reshape(121,1)
x_y=np.concatenate((x_array1,y_array1),axis=1)
xy_array=x_y=np.concatenate((x_array1,y_array1),axis=1)


# defining a Linear neural network
xy_train, xy_test, u_train, u_test = train_test_split(xy_array, u_array, test_size=0.20, random_state=40)
lr = LinearRegression()
lr.fit(xy_train, u_train)
pred_train_lr= lr.predict(xy_train)
print(np.sqrt(mean_squared_error(u_train,pred_train_lr)))
print(r2_score(u_train, pred_train_lr))
pred_test_lr= lr.predict(xy_test)
print(np.sqrt(mean_squared_error(u_test,pred_test_lr)))
print(r2_score(u_test, pred_test_lr))

#new prediction for high-resolution grid
u1= fenics_solver(nx=200)
mesh=UnitSquareMesh(200, 200)
W=FunctionSpace(mesh, "CG", 1)
u1_array=fenics_to_numpy(u1)
x=fenics.interpolate(Expression("x[0]",degree=1),W)
x_array=fenics_to_numpy(x)
x_array1= x_array.reshape(40401,1)
y=fenics.interpolate(Expression("x[1]",degree=1),W)
y_array=fenics_to_numpy(y)
y_array1=y_array.reshape(40401,1)
xy=np.concatenate((x_array1,y_array1),axis=1)
pred_new_lr= lr.predict(xy)
print(np.shape(pred_new_lr.shape))
print(np.sqrt(mean_squared_error(u1_array,pred_new_lr)))
print(r2_score(u1_array, pred_new_lr))


# plotting the results
u=numpy_to_fenics(pred_new_lr,fenics.Function(W))
plot(u,title="LinearNN_Solution")
plot(u1,title="Fenics_Solution")
plt.show()
