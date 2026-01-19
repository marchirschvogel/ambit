#!/usr/bin/env python3

"""
A 2D fluid dynamics problem of incompressible Navier-Stokes channel flow around a rigid cylindrical obstacle.
"""

import ambit_fe
import numpy as np
from pathlib import Path


def main():
    basepath = str(Path(__file__).parent.absolute())

    """
    Parameters for input/output
    """
    IO_PARAMS = {  # problem type 'fluid': incompressible Navier-Stokes flow
        "problem_type": "fluid",
        # the meshes for the domain and boundary topology are specified separately
        "mesh_domain": basepath + "/input/channel_domain.xdmf",
        "mesh_boundary": basepath + "/input/channel_boundary.xdmf",
        # at which step frequency to write results (set to 0 in order to not write any output)
        "write_results_every": 1,
        # where to write the output to
        "output_path": basepath + "/tmp/",
        # which results to write
        "results_to_write": ["velocity", "pressure"],
        # the 'midfix' for all simulation result file names: will be results_<simname>_<field>.xdmf/.h5
        "simname": "fluid_channel",
    }

    """
    Parameters for the global time control
    """
    CONTROL_PARAMS = {"maxtime": 0.5, "numstep": 100}

    """
    Parameters for the linear and nonlinear solution schemes
    """
    SOLVER_PARAMS = {
        "solve_type": "direct",
        #'direct_solver'         : 'superlu_dist', # no idea why, but mumps does not seem to like this system in parallel...
        "tol_res": [1e-8, 1e-8],
        "tol_inc": [1e-8, 1e-8],
    }

    """
    Parameters for the fluid mechanics time integration scheme
    """
    TIME_PARAMS = {"timint": "ost", "theta_ost": 1.0}

    """
    Finite element parameters: Taylor-Hood elements with quadratic approximation for the velocity and linear approximation for the pressure
    """
    FEM_PARAMS = {  # the order of the finite element ansatz functions for the velocity and pressure
        "order_vel": 2,
        "order_pres": 1,
        # the degree of the quadrature scheme
        "quad_degree": 5,
    }

    """
    Constitutive parameters for the fluid: let's use parameters of glycerine
    """
    MATERIALS = {
        "MAT1": {
            "newtonian": {"eta": 1420.0e-6},  # kPa s
            "inertia": {"rho": 1.26e-6},
        }
    }  # kg/mm^3

    """
    Time curves, e.g. any prescribed time-controlled/-varying loads or functions
    """

    class time_curves:
        def tc1(self, t):
            Umax = 1e3

            t_ramp = 0.3

            return Umax * 0.5 * (1.0 - np.cos(np.pi * t / t_ramp)) * (t < t_ramp) + Umax * (t >= t_ramp)

    """
    Boundary conditions: ids: 1: inflow, 2: bottom wall, 3: axial outflow, 4: top wall - 5: obstacle
    Wall and obstactle are fixed, in-flow velocity is prescribed, outflow is free ("Neumann zero")
    """
    BC_DICT = {
        "dirichlet": [
            {"id": [1], "dir": "x", "curve": 1},
            {"id": [2, 4, 5], "dir": "all", "val": 0.0},
        ]
    }

    # problem setup
    problem = ambit_fe.ambit_main.Ambit(
        IO_PARAMS,
        CONTROL_PARAMS,
        TIME_PARAMS,
        SOLVER_PARAMS,
        FEM_PARAMS,
        MATERIALS,
        BC_DICT,
        time_curves=time_curves(),
    )

    # solve time-dependent problem
    problem.solve_problem()


if __name__ == "__main__":
    main()
