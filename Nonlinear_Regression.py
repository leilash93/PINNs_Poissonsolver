from dolfin import *
import fenics
import pyadjoint
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
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


# defining a neural network using the Decision Trees algorithm
xy_train, xy_test, u_train, u_test = train_test_split(xy_array, u_array, test_size=0.20, random_state=40)
print(xy_train.shape); print(xy_test.shape)
dtree = DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=3)
dtree.fit(xy_train, u_train)
pred_train_tree= dtree.predict(xy_train)
print(np.sqrt(mean_squared_error(u_train,pred_train_tree)))
print(r2_score(u_train, pred_train_tree))
pred_test_tree= dtree.predict(xy_test)
print(np.sqrt(mean_squared_error(u_test,pred_test_tree)))
print(r2_score(u_test, pred_test_tree))


# new prediction for high resolution grid
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
pred_new_tree= dtree.predict(xy)
print(np.sqrt(mean_squared_error(u1_array,pred_new_tree)))
print(r2_score(u1_array, pred_new_tree))

# defining a neural network using the Random Forest algorithm
model_rf = RandomForestRegressor(n_estimators=10, oob_score=True, random_state=10)
model_rf.fit(xy_train, u_train)
pred_train_rf= model_rf.predict(xy_train)
print(np.sqrt(mean_squared_error(u_train,pred_train_rf)))
print(r2_score(u_train, pred_train_rf))
pred_test_rf = model_rf.predict(xy_test)
print(np.sqrt(mean_squared_error(u_test,pred_test_rf)))
print(r2_score(u_test, pred_test_rf))
pred_new_rf= model_rf.predict(xy)
print(np.sqrt(mean_squared_error(u1_array,pred_new_rf)))
print(r2_score(u1_array, pred_new_rf))


# plotting the results
u=numpy_to_fenics(pred_new_tree,fenics.Function(W))
u1=numpy_to_fenics(pred_new_rf,fenics.Function(W))
plot(u,title="TreeNN_Solution")
plot(u1,title="ForestNN_Solution")
plt.show()