import fenics
import numpy as np

def numpy_to_fenics(numpy_array, fenics_var_template):
#Convert numpy array to FEniCS variable

    if isinstance(fenics_var_template, fenics.Constant):
        if numpy_array.shape == (1,):
            return type(fenics_var_template)(numpy_array[0])
        else:
            return type(fenics_var_template)(numpy_array)

    if isinstance(fenics_var_template, fenics.Function):
        function_space = fenics_var_template.function_space()

        u = type(fenics_var_template)(function_space)

        # assume that given numpy array is global array that needs to be distrubuted across processes
        # when FEniCS function is created
        fenics_size = u.vector().size()
        np_size = numpy_array.size

        if np_size != fenics_size:
            err_msg = (
                f"Cannot convert numpy array to Function:"
                f"Wrong size {numpy_array.size} vs {u.vector().size()}"
            )
            raise ValueError(err_msg)

        if numpy_array.dtype != np.float_:
            err_msg = (
                f"The numpy array must be of type {np.float_}, "
                "but got {numpy_array.dtype}"
            )
            raise ValueError(err_msg)

        range_begin, range_end = u.vector().local_range()
        numpy_array = np.asarray(numpy_array)
        local_array = numpy_array.reshape(fenics_size)[range_begin:range_end]
        u.vector().set_local(local_array)
        u.vector().apply("insert")
        return u