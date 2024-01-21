#!/usr/bin/env python3

"""
FSI of elastic flag in channel (2D) (modified Turek benchmark): Q2-Q1 Taylor-Hood for both fluid and incompressible solid
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
                            'indicate_results_by'   : 'step0',
                            # where to write the output to
                            'output_path'           : basepath+'/tmp/',
                            'mesh_domain'           : basepath+'/input/channel-flag_domain.xdmf',
                            'mesh_boundary'         : basepath+'/input/channel-flag_boundary.xdmf',
                            'results_to_write'      : [['displacement','velocity'], [['fluiddisplacement','velocity','pressure'],['aledisplacement','alevelocity']]],
                            'domain_ids_solid'      : [1], 
                            'domain_ids_fluid'      : [2],
                            'surface_ids_interface' : [1],
                            'simname'               : 'test_parabinflow'}

    """
    Parameters for the linear and nonlinear solution schemes
    """
    SOLVER_PARAMS        = {'solve_type'            : 'direct',
                            'direct_solver'         : 'mumps',
                            'divergence_continue'   : 'PTC',
                            'k_ptc_initial'         : 10.,
                            # residual and increment tolerances
                            'tol_res'               : [1e-8,1e-8,1e-8,1e-8,1e-8,1e-3],
                            'tol_inc'               : [1e-0,1e-0,1e-0,1e-0,1e10,1e-0]}

    """
    Parameters for the solid mechanics time integration scheme, plus the global time parameters
    """
    TIME_PARAMS_SOLID    = {'maxtime'               : 35.0,
                            'numstep'               : 8750,
                            #'numstep_stop'          : 0,
                            'timint'                : 'ost',
                            'theta_ost'             : 1.0}

    """
    Parameters for the fluid mechanics time integration scheme, plus the global time parameters
    """
    TIME_PARAMS_FLUID    = {'maxtime'               : 35.0,
                            'numstep'               : 8750,
                            #'numstep_stop'          : 0,
                            'timint'                : 'ost',
                            'theta_ost'             : 1.0}

    """
    Finite element parameters for solid: Taylor-Hood space
    """
    FEM_PARAMS_SOLID     = {'order_disp'            : 2,
                            'order_pres'            : 1,
                            'quad_degree'           : 5,
                            'incompressible_2field' : True}

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
                            'zero_lm_boundary'      : False, # TODO: Seems to select the wrong dofs on LM mesh! Do not turn on!
                            'fluid_on_deformed'     : 'consistent'}

    # parameters for polybutadiene (Tab. 2 Turek et al. 2006)
    MATERIALS_SOLID      = {'MAT1' : {'neohooke_dev'      : {'mu' : 0.53e3},
                                      'inertia'           : {'rho0' : 0.91e-6}}}

    # parameters for glycerine (Tab. 2 Turek et al. 2006)
    MATERIALS_FLUID      = {'MAT1' : {'newtonian' : {'mu' : 1420.0e-6},
                                      'inertia' : {'rho' : 1.26e-6}}}
    
    # linear elastic material for domain motion problem
    MATERIALS_ALE        = {'MAT1' : {'linelast' : {'Emod' : 2.0, 'kappa' : 10.}}}


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
    BC_DICT_SOLID        = { 'dirichlet' : [{'id' : [6], 'dir' : 'all', 'val' : 0.}]}

    BC_DICT_FLUID        = { 'dirichlet' : [{'id' : [4], 'dir' : 'x', 'expression' : expression1},
                                            {'id' : [2,3], 'dir' : 'all', 'val' : 0.}],
                            'stabilized_neumann' : [{'id' : [5], 'par1' : 0.252e-6, 'par2' : 1.}] }

    BC_DICT_ALE          = { 'dirichlet' : [{'id' : [2,3,4,5], 'dir' : 'all', 'val' : 0.}] }


    # Pass parameters to Ambit to set up the problem
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, [TIME_PARAMS_SOLID, TIME_PARAMS_FLUID], SOLVER_PARAMS, [FEM_PARAMS_SOLID, FEM_PARAMS_FLUID, FEM_PARAMS_ALE], [MATERIALS_SOLID, MATERIALS_FLUID, MATERIALS_ALE], [BC_DICT_SOLID, BC_DICT_FLUID, BC_DICT_ALE], time_curves=time_curves, coupling_params=COUPLING_PARAMS)

    # Call the Ambit solver to solve the problem
    problem.solve_problem()



if __name__ == "__main__":

    main()
