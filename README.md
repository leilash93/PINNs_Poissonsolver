# Solving Non-linear Poisson with Physics Informed Neural Networks (PINNS)
The following sample problem will demonstrate that deep neural network can be utilized to approximate the solutions of partial differential equations (PDEs). This is a recent area of research, known as Physics Informed Neural Networks (PINNS). With this approach, a loss function is setup to penalize the fitted function’s deviation from the desired differential operator and boundary conditions. ﻿The main insight of this approach lies in the fact that the training data consists of randomly sampled points in the function’s domain. By sampling mini-batches from different parts of the domain and processing these small batches sequentially, the neural network “learns” the function without the computational bottleneck present with grid-based methods. 

a non-linear Poisson equation:

∇.((1+u^2 )∇u)=f in Ω

f=xsin(y)

u=1 on ∂Ω_D

∇u.n=0 on ∂Ω_N

Developer:
- Leila Shahsavari

n this project we are trying to s...
Here is the reference articles stimulated this project design:
1. Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations, 2018,
	Basant Agarwala,b, Heri Ramampiaroa, Helge Langsetha, Massimiliano Ruocco
2. Siamese Capsule Networks with Global and Local Features for Text Classification, 2020,
	Yujia Wu, Jing Li, Jia Wu, Jun Chang
  
3. Hybrid FEM-NN models: Combining artificial neural networks with the fnite element method, 2021,
  Sebastian K. Mituscha, Simon W. Funkea, Miroslav Kucht
  
4. Solving PDEs in Python– The FEniCS Tutorial Volume I, 2017,
  Hans Petter Langtange, Anders Logg


Libraries:

fenics

matplotlib

pyadjoint

scikit-learn

numpy

pandas
