from dolfin import *
import pandas as pd
import matplotlib.pyplot as plt

# defining fenics solver
def fenics_solver(nx):
    if has_linear_algebra_backend("Epetra"):
        parameters["linear_algebra_backend"] = "Epetra"

        # Subdomain for Dirichlet boundary condition
    class DirichletBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return abs(x[0] - 1.0) < DOLFIN_EPS and on_boundary
    # Create mesh and define function space
    mesh = UnitSquareMesh(nx, nx)
    V = FunctionSpace(mesh, "CG", 1)
    # Define boundary condition
    g = Constant(1.0)
    bc = DirichletBC(V, g, DirichletBoundary())

    # Define variational problem
    u = Function(V)
    v = TestFunction(V)
    f = Expression("x[0]*sin(x[1])", degree=2)
    F = inner((1 + u ** 2) * grad(u), grad(v)) * dx - f * v * dx

    # Compute solution
    solve(F == 0, u, bc, solver_parameters={"newton_solver":
                                                            {"relative_tolerance": 1e-6}})
    return u

#--------------------------------------------------------------------------------------------------------------

# exact solution in fenics
u_e=fenics_solver(nx=200)
# finding error for nx=5,10,20,40
rows=[]
for nx in [5,10,20,40]:
    u= fenics_solver(nx)
    E= errornorm(u,u_e,norm_type='L2')
    rows.append([nx,E])
df=pd.DataFrame(rows,columns=["nx","Error"])
print(df)
plt.plot(df['nx'],df['Error'],marker='o')
plt.title('L2-Error For Different Meshes')
plt.xlabel('Mesh Size (nx)')
plt.ylabel('Error')
plt.show()
