import fenics
import pyadjoint
import numpy as np
from typing import Union
FenicsVariable = Union[fenics.Constant, fenics.Function]

def fenics_to_numpy(fenics_var):
#Convert FEniCS variable to numpy array.
    if isinstance(fenics_var, fenics.Constant):
        return np.asarray(fenics_var.values())

    if isinstance(fenics_var, fenics.Function):
        fenics_vec = fenics_var.vector()
        if fenics_vec.mpi_comm().size > 1:
            data = fenics_vec.gather(np.arange(fenics_vec.size(), dtype="I"))
        else:
            data = fenics_vec.get_local()
        return np.asarray(data)

    if isinstance(fenics_var, fenics.GenericVector):
        if fenics_var.mpi_comm().size > 1:
            data = fenics_var.gather(np.arange(fenics_var.size(), dtype="I"))
        else:
            data = fenics_var.get_local()
        return np.asarray(data)

    if isinstance(fenics_var, (pyadjoint.AdjFloat, float)):
        return np.asarray(fenics_var)

    raise ValueError("Cannot convert " + str(type(fenics_var)))