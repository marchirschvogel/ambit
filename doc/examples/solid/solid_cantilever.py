#!/usr/bin/env python3

"""
A quasi-static cantilever under transverse load.
"""

import ambit_fe
import numpy as np
from pathlib import Path

def main():

    basepath = str(Path(__file__).parent.absolute())


    """
    Parameters for input/output
    """
    IO_PARAMS         = {# problem type 'solid': a finite strain elastostatic or elastodynamic analysis
                         'problem_type'          : 'solid',
                         # the meshes for the domain and boundary topology are specified separately
                         'mesh_domain'           : basepath+'/input/cantilever_domain.xdmf',
                         'mesh_boundary'         : basepath+'/input/cantilever_boundary.xdmf',
                         # at which step frequency to write results (set to 0 in order to not write any output)
                         'write_results_every'   : 1,
                         # where to write the output to
                         'output_path'           : basepath+'/tmp/',
                         # which results to write: here, we want the displacement as well as the von Mises Cauchy stress
                         'results_to_write'      : ['displacement','vonmises_cauchystress'],
                         # the 'midfix' for all simulation result file names: will be results_<simname>_<field>.xdmf/.h5
                         'simname'               : 'solid_cantilever'}

    """
    Parameters for the linear and nonlinear solution schemes
    """
    SOLVER_PARAMS     = {# this specifies which linear solution strategy to use; since this problem has less than 3'000 degrees of freedom, we comfortably can use a direct solver
                         'solve_type'            : 'direct',
                         # residual and increment tolerances
                         'tol_res'               : 1.0e-4, # N
                         'tol_inc'               : 1.0e-8} # m

    """
    Parameters for the solid mechanics time integration scheme, as well as the global time parameters
    """
    TIME_PARAMS       = {'maxtime'               : 1.0,
                         'numstep'               : 100,
                         'timint'                : 'static'}

    """
    Finite element parameters
    """
    FEM_PARAMS        = {# the order of the finite element ansatz functions for the displacement: our input mesh is quadratic, so we should use 2nd order here
                         'order_disp'            : 2,
                         # the degree of the quadrature rule
                         'quad_degree'           : 6,
                         # whether to run fully incompressible: here we have a steel cantilever, so far from being incompressible
                         'incompressible_2field' : False}

    """
    Constitutive parameters: assume the cantilever is made of steel
    """
    MATERIALS         = {# we use the St. Venant-Kirchhoff material here, which should be used only for small (compressive) strains but is large rotation capable
                         # (so a generalization of Hooke's law for linear elasticity)
                         'MAT1' : {'stvenantkirchhoff' : {'Emod' : 210e9, 'nu' : 0.3}, # Pa
                                   # the inertia: density - NOTE: since we're doing a quasi-static analysis, we might as well remove the inertia entry since it's not used.
                                   'inertia'           : {'rho0' : 7850.}}} # kg/m^3


    """
    Time curves, e.g. any prescribed time-controlled/-varying loads or functions
    """
    class time_curves():

        def tc1(self, t): # curve controlling transversal load, here linearly ramped from 0 to load
            load = 1e9 # Pa
            return load*t/TIME_PARAMS['maxtime']

    """
    Boundary conditions: The cantilever is fixed on one end (Dirichlet condition) and transversally loaded with a convervative (PK1) Neumann traction, ramped by curve no. 1
    """
    BC_DICT           = { 'dirichlet' : [{'id' : [1], 'dir' : 'all', 'val' : 0.0}],
                            'neumann' : [{'id' : [2], 'dir' : 'xyz_ref', 'curve' : [0,1,0]}] }


    # Pass parameters to Ambit to set up the problem
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves())

    # Call the Ambit solver to solve the problem
    problem.solve_problem()




if __name__ == "__main__":

    main()
