#!/usr/bin/env python3

"""
FSI of elastic flag in channel (2D) (Turek benchmark): Q2-Q1 Taylor-Hood elements for fluid and Q2 elements for solid
Reference solution: https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_reference.html
"""

import ambit_fe
import numpy as np
from pathlib import Path


def main():

    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try: restart_step = int(sys.argv[1])
    except: restart_step = 0

    """
    Parameters for input/output
    """
    IO_PARAMS            = {# problem type 'fsi': fluid-solid interaction
                            'problem_type'          : 'fsi',
                            'USE_MIXED_DOLFINX_BRANCH' : True,
                            # at which step frequency to write results (set to 0 in order to not write any output): here, only every 10th due to the many steps
                            'write_results_every'   : 10,
                            'write_restart_every'   : -1,
                            'restart_step'          : restart_step,
                            # where to write the output to
                            'output_path'           : basepath+'/tmp/',
                            'mesh_domain'           : basepath+'/input/channel-flag_domain.xdmf',
                            'mesh_boundary'         : basepath+'/input/channel-flag_boundary.xdmf',
                            'results_to_write'      : [['displacement','velocity'], [['fluiddisplacement','velocity','pressure'],['aledisplacement','alevelocity']]],
                            'domain_ids_solid'      : [1], 
                            'domain_ids_fluid'      : [2],
                            'surface_ids_interface' : [1],
                            'simname'               : 'fsi_channel_flag_turekFSI2'}

    """
    Parameters for the linear and nonlinear solution schemes
    """
    SOLVER_PARAMS        = {'solve_type'            : 'direct',
                            'direct_solver'         : 'mumps',
                            # residual and increment tolerances
                            'tol_res'               : [1e-8,1e-8,1e-8,1e-8,1e-3], # solid-mom,fluid-mom,fluid-cont,FSI-coup,ALE-mom
                            'tol_inc'               : [1e-0,1e-0,1e-0,1e10,1e-0]} # du,dv,dp,dlm,dd

    """
    Parameters for the solid mechanics time integration scheme, plus the global time parameters
    """
    TIME_PARAMS_SOLID    = {'maxtime'               : 35.0,
                            'numstep'               : 8750, # 8750: dt=0.004 s - 17500: dt=0.002 s - 35000: dt=0.001 s
                            #'numstep_stop'          : 0,
                            'timint'                : 'ost',
                            'theta_ost'             : 0.5, # 0.5: Crank-Nicholson, 1.0: Backward Euler
                            # how to evaluat nonlinear terms f(x) in the midpoint time-integration scheme:
                            # trapezoidal: theta * f(x_{n+1}) + (1-theta) * f(x_{n})
                            # midpoint:    f(theta*x_{n+1} + (1-theta)*x_{n})
                            'eval_nonlin_terms'     : 'midpoint'} # trapezoidal, midpoint

    """
    Parameters for the fluid mechanics time integration scheme, plus the global time parameters
    """
    TIME_PARAMS_FLUID    = {'maxtime'               : 35.0,
                            'numstep'               : 8750, # 8750: dt=0.004 s - 17500: dt=0.002 s - 35000: dt=0.001 s
                            #'numstep_stop'          : 0,
                            'timint'                : 'ost',
                            'theta_ost'             : 0.5, # 0.5: Crank-Nicholson, 1.0: Backward Euler
                            # how to evaluate nonlinear terms f(x) in the midpoint time-integration scheme:
                            # trapezoidal: theta * f(x_{n+1}) + (1-theta) * f(x_{n})
                            # midpoint:    f(theta*x_{n+1} + (1-theta)*x_{n})
                            'eval_nonlin_terms'     : 'midpoint'} # trapezoidal, midpoint

    """
    Finite element parameters for solid: Taylor-Hood space
    """
    FEM_PARAMS_SOLID     = {'order_disp'            : 2,
                            'order_pres'            : 1,
                            'quad_degree'           : 5,
                            'incompressible_2field' : False}

    """
    Finite element parameters for fluid: Taylor-Hood space
    """
    FEM_PARAMS_FLUID     = {'order_vel'             : 2,
                            'order_pres'            : 1,
                            'quad_degree'           : 5}
    
    """
    Finite element parameters for ALE
    """
    FEM_PARAMS_ALE       = {'order_disp'            : 2,
                            'quad_degree'           : 5}
    
    """
    FSI coupling parameters
    """
    COUPLING_PARAMS      = {'coupling_fluid_ale'    : [{'surface_ids' : [1], 'type' : 'strong_dirichlet'}],
                            'fsi_governing_type'    : 'solid_governed', # solid_governed, fluid_governed
                            'zero_lm_boundary'      : False} # TODO: Seems to select the wrong dofs on LM mesh! Do not turn on!

    # parameters for FSI2 case (Tab. 12 Turek et al. 2006)
    MATERIALS_SOLID      = {'MAT1' : {'stvenantkirchhoff' : {'Emod' : 1.4e3, 'nu' : 0.4}, # kPa, E = 2*mu*(1+nu)
                                      'inertia'           : {'rho0' : 10.0e-6}}} # kg/mm^3

    # parameters for FSI2 case (Tab. 12 Turek et al. 2006)
    MATERIALS_FLUID      = {'MAT1' : {'newtonian' : {'mu' : 1.0e-3}, # kPa s
                                      'inertia' : {'rho' : 1.0e-6}}} # kg/mm^3
    
    # linear elastic material for domain motion problem
    MATERIALS_ALE        = {'MAT1' : {'neohooke' : {'mu' : 10.0, 'nu' : 0.3}}}


    """
    User expression, here a spatially varying time-controlled inflow: always need a class variable self.t and an evaluate(self, x)
    with the only argument being the spatial coordinates x
    """
    class expression1:
        def __init__(self):
            
            self.t = 0.0

            self.t_ramp = 2.0
            
            self.H = 0.41e3 # channel height
            self.Ubar = 1e3

        def evaluate(self, x):
            
            vel_inflow_y = 1.5*self.Ubar*( x[1]*(self.H-x[1])/((self.H/2.)**2.) )

            val_t = vel_inflow_y * 0.5*(1.-np.cos(np.pi*self.t/self.t_ramp)) * (self.t < self.t_ramp) + vel_inflow_y * (self.t >= self.t_ramp)

            return ( np.full(x.shape[1], val_t),
                     np.full(x.shape[1], val_t) )


    """
    Boundary conditions
    """
    BC_DICT_SOLID        = { 'dirichlet' : [{'id' : [6], 'dir' : 'all', 'val' : 0.}] }

    BC_DICT_FLUID        = { 'dirichlet' : [{'id' : [4], 'dir' : 'x', 'expression' : expression1},
                                            {'id' : [2,3], 'dir' : 'all', 'val' : 0.}] }

    BC_DICT_ALE          = { 'dirichlet' : [{'id' : [2,3,4,5], 'dir' : 'all', 'val' : 0.}] }


    # Pass parameters to Ambit to set up the problem
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, [TIME_PARAMS_SOLID, TIME_PARAMS_FLUID], SOLVER_PARAMS, [FEM_PARAMS_SOLID, FEM_PARAMS_FLUID, FEM_PARAMS_ALE], [MATERIALS_SOLID, MATERIALS_FLUID, MATERIALS_ALE], [BC_DICT_SOLID, BC_DICT_FLUID, BC_DICT_ALE], coupling_params=COUPLING_PARAMS)

    # Call the Ambit solver to solve the problem
    problem.solve_problem()



if __name__ == "__main__":

    main()
