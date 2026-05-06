#!/usr/bin/env python3

"""
Multiphase FSI elaso-capillary simulation of a sessile droplet on a soft solid substrate
Example from M. Shokrpour Roudbari and E. H. van Brummelen, "Binary-Fluid-Solid Interaction Based on the Navier-Stokes-Korteweg Equations", Mathematical Models and Methods in Applied Sciences, 2019
with a more recent version in E. H. van Brummelen et al. "An adaptive isogeometric analysis approach to elasto-capillary fluid-solid interaction", International Journal for Numerical Methods in Engineering, 2021
Cases (1,2,3) taken from the latter publication!
Let's assume a pg-µm-µs unit system
"""

import ambit_fe
import numpy as np
from pathlib import Path


def main():
    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try:
        restart_step = int(sys.argv[1])
    except:
        restart_step = 0

    case = 1

    dim = "2D" # 2D, 3D - need to set it up for 3D appropriately (mesh...)

    """
    Refinement strip around elastocapillary contact region
    """
    class locate_refine_strip:
        def evaluate(self, x):
            return np.logical_and(x[0] > 120., x[0] < 230.)

    IO_PARAMS = {
        "problem_type": "fsi_multiphase",
        "write_results_every": 10,
        "write_restart_every": -1,
        "indicate_results_by": "step",
        "restart_step": restart_step,
        "output_path": basepath + "/tmp/",
        "mesh_domain": {"type":"rectangle", "celltype":"triangle", "coords_a":[0.0, -50.0], "coords_b":[500.0, 500.0], "meshsize":[40,44]},
        # "mesh_domain": {"type":"box", "celltype":"tetrahedron", "coords_a":[0.0, -50.0, 0.0], "coords_b":[500.0, 500.0, 500.0], "meshsize":[40,44,40]},
        "mesh_encoding": "ASCII", # HDF5, ASCII
        "refine_mesh": {"region": locate_refine_strip(), "steps": 3},  # refinement working only for triangles/tetrahedra
        "results_to_write": [
            ["displacement"],
            ["velocity", "pressure", "density"],
            ["phase", "potential"],
            ["aledisplacement"],
        ],
        "write_initial_fields": True,
        "report_conservation_properties": True,
        # "write_submeshes": True,
        "simname": "fsi_multiphase_elastocapillary"+str(case)+"_"+dim+"R3",
    }

    eps = 1.0 # 1 µm (E. H. van Brummelen et al. 2021)

    class expr1:
        def __init__(self):
            self.t = 0
            self.R_0 = 178.4 # µm
            self.x_c = np.asarray([0.0, 19.4, 0.0])

        def evaluate(self, x):
            d = np.sqrt( (x[0]-self.x_c[0])**2.0 + (x[1]-self.x_c[1])**2.0 + (x[2]-self.x_c[2])**2.0 )
            val = np.tanh((self.R_0 - d)/(np.sqrt(2.0)*eps))  # phi in [-1,1]
            return (
                np.full(x.shape[1], val),
            )

    # fluid1 is vapour (surrounding), fluid2 is liquid (bubble)
    if case==1:
        dt = 0.01e3#0.1e3 # µs
        rho1 = 1.26 #0.0816 # pg/(µm^3) = pg/(µm^3)
        rho2 = 1.26 #0.2408 # pg/(µm^3) = pg/(µm^3)
        eta1 = 1412. # 1 mPa s = pg/(µm µs)
        eta2 = 1412. # Glycerol - 1 mPa s = pg/(µm µs)
        sig = 46. # surface energy density coefficient - mN/m = g/(s^2) = pg/(µs^2)
        rho_s = 1.0#12.6e-3 # solid density - 1 pg/(µm^3) = pg/(µm^3)
    elif case==2:
        dt = 0.01e3 # µs
        rho1 = 0.1 # 1 pg/(µm^3) = pg/(µm^3)
        rho2 = 1.26 # 1 pg/(µm^3) = pg/(µm^3)
        eta1 = 100. # 1 mPa s = pg/(µm µs)
        eta2 = 1412. # Glycerol - 1 mPa s = pg/(µm µs)
        sig = 46. # surface energy density coefficient - mN/m = g/(s^2) = pg/(µs^2)
        rho_s = 1.0 # solid density - 1 pg/(µm^3) = pg/(µm^3)
    else:
        raise ValueError("Unknown case.")

    """
    Parameters for the global time control
    """
    CONTROL_PARAMS = {"maxtime": 10e3, # µs
                      "dt": dt,
                      # "numstep_stop": 100,
                      "initial_fields": [expr1, None],
                      }

    SOLVER_PARAMS = {
        "solve_type": "direct",
        "direct_solver": "mumps",   # superlu_dist, mumps
        "tol_res": 1e-6,
        "tol_inc": 1e-4,
    }

    TIME_PARAMS_SOLID = {"timint": "genalpha", "rho_inf_genalpha": 0.8, "eval_nonlin_terms": "trapezoidal"}
    TIME_PARAMS_FLUID = {"timint": "bdf2"}
    TIME_PARAMS_PF    = {"timint": "bdf2"}

    E = 3.0 # 1 kPa = pg/(µm µs^2)
    nu = 0.499

    FEM_PARAMS_SOLID = {
        "order_disp": 2,
        "order_pres": 1,
        "quad_degree": 5,
        "incompressibility": "no",
        "bulkmod": E/(3.*(1.-2.*nu)),
    }

    FEM_PARAMS_FLUID = {"order_vel": 2,
                        "order_pres": 1,
                        "quad_degree": 5,
                        "fluid_formulation": "conservative"}

    FEM_PARAMS_ALE = {"order_disp": 2, "quad_degree": 5}

    FEM_PARAMS_PF = {"order_phi": 1, "order_mu": 1, "quad_degree": 5}

    class locate_interf:
        def evaluate(self, x):
            return np.isclose(x[1], 0.0)

    sig_sl = 36.
    sig_sa = 31.
    COUPLING_PARAMS_FSI = {
        "coupling_fluid_ale": {"interface": [locate_interf()]},
        "fsi_system": "neumann_dirichlet",  # neumann_neumann, neumann_dirichlet
        "wetting_condition_interface": {"coeff": 3.*(sig_sa-sig_sl)/4.}, # wetting Robin condition at interface
    }

    # Use full Korteweg stress in capillary force contribution - needed for correct inclusion of capillary traction forces at FSI interface!
    COUPLING_PARAMS_MULTIPHASE = {"capillary_force_from_korteweg_stress": True,
                                  "clip_phi_range": True,
                                  "smooth_clip": "cubic",
                                  "epsilon_clip": 1e-3}

    dlt=1e-5
    class locate_solid:
        def evaluate(self, x):
            return (x[1] <= 0.0+dlt)

    class locate_fluid:
        def evaluate(self, x):
            return (x[1] >= 0.0-dlt)

    # locators for boundary conditions
    class locate_right:
        def evaluate(self, x):
            return np.isclose(x[0], 500.0)
    class locate_left:
        def evaluate(self, x):
            return np.isclose(x[0], 0.0)

    class locate_top:
        def evaluate(self, x):
            return np.isclose(x[1], 500.0)

    class locate_bottom:
        def evaluate(self, x):
            return np.isclose(x[1], -50.0)

    # for 3D
    class locate_back:
        def evaluate(self, x):
            return np.isclose(x[2], 0.0)
    class locate_front:
        def evaluate(self, x):
            return np.isclose(x[2], 500.0)


    zeta = 0.0

    MATERIALS_SOLID = {"MAT1": {"neohooke_dev": {"mu": E/3.},
                                "ogden_vol": {"kappa": E/(3.*(1.-2.*nu))},
                                "inertia": {"rho0": rho_s},
                                "id": locate_solid()}}

    alpha = (rho1-rho2)/(rho1+rho2)
    sigtilde = sig #3.*sig/(2.*np.sqrt(2.))

    MATERIALS_FLUID = {"MAT1": {"newtonian": {"eta1": eta1, "eta2": eta2, "zeta1": zeta, "zeta2": zeta},
                                "inertia": {"rho1": rho1, "rho2": rho2},
                                "id": locate_fluid()}}

    # MATERIALS_ALE = {"MAT1": {"neohooke": {"mu": 10.0, "nu": 0.3, "scale_det" : True, "scale_exp" : 1.5}, "id": locate_fluid()}}
    MATERIALS_ALE = {"MAT1": {"exponential": {"a_0": 1.0, "b_0": 10.0, "kappa": 1e2}, "id": locate_fluid()}}

    m = 1e-5 # should be rather low if capillary stress is rather high
    MATERIALS_PF = {"MAT1": {"mat_cahnhilliard": {"mobility": "constant", # constant, degenerate
                                                  "epsilon": 0.0,
                                                  "exponent": 1.0,
                                                  # "M0": m*eps**2.0,      # Mobility [length^5/(pressure time)]
                                                  "M0": m,                 # Mobility [length^5/(pressure time)]
                                                  "D": sigtilde/(4.*eps),  # Bulk free-energy parameter [pressure/length^3]
                                                  "kappa": sigtilde*eps,   # Gradient energy coefficient [pressure/length]
                                                  "alpha": None},         # Pressure factor in diffusive flux
                                                  "id": locate_fluid()}}

    class locate_corner:
        def evaluate(self, x):
            ctr_x = np.isclose(x[0], 0.0)
            ctr_y = np.isclose(x[1], 0.0)
            return np.logical_and(ctr_x, ctr_y)


    BC_DICT_SOLID = {
        "dirichlet": [{"id": [locate_bottom()], "dir": "all", "val": 0.0},
                      {"id": [locate_left(),locate_right()], "dir": "x", "val": 0.0}],
        }

    BC_DICT_FLUID = {
        "dirichlet": [{"id": [locate_top()], "dir": "all", "val": 0.0},
                      {"id": [locate_left(),locate_right()], "dir": "x", "val": 0.0}],
        # "dirichlet_pres" : [{"id": [locate_corner()], "dir": "all", "val": 0.0}],
    }

    BC_DICT_ALE = {
        "dirichlet": [{"id": [locate_top()], "dir": "all", "val": 0.0},
                      {"id": [locate_left(),locate_right()], "dir": "x", "val": 0.0}],
    }

    BC_DICT_PF = { }# "dirichlet": [{"id": [locate_top(),locate_right()], "dir": "all", "val": -1.0}] }


    BC_DICT_LM = {"dirichlet": [{"id": [locate_left(),locate_right()], "dir": "x", "val": 0.0}]}

    if dim=="3D":
        BC_DICT_SOLID["dirichlet"].append({"id": [locate_front(),locate_back()], "dir": "z", "val": 0.0})
        BC_DICT_FLUID["dirichlet"].append({"id": [locate_front(),locate_back()], "dir": "z", "val": 0.0})
        BC_DICT_ALE["dirichlet"].append({"id": [locate_front(),locate_back()], "dir": "z", "val": 0.0})
        BC_DICT_LM["dirichlet"].append({"id": [locate_front(),locate_back()], "dir": "z", "val": 0.0})


    problem = ambit_fe.ambit_main.Ambit(
        IO_PARAMS,
        CONTROL_PARAMS,
        [TIME_PARAMS_SOLID, TIME_PARAMS_FLUID, TIME_PARAMS_PF],
        SOLVER_PARAMS,
        [FEM_PARAMS_SOLID, FEM_PARAMS_FLUID, FEM_PARAMS_PF, FEM_PARAMS_ALE],
        [MATERIALS_SOLID, MATERIALS_FLUID, MATERIALS_PF, MATERIALS_ALE],
        [BC_DICT_SOLID, BC_DICT_FLUID, BC_DICT_PF, BC_DICT_ALE, BC_DICT_LM],
        coupling_params=[COUPLING_PARAMS_FSI,COUPLING_PARAMS_MULTIPHASE],
    )

    # problem solve
    problem.solve_problem()


if __name__ == "__main__":
    main()
