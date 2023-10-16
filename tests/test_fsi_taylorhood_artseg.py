#!/usr/bin/env python3

"""
FSI of arterial segment: Q2-Q1 Taylor-Hood for both fluid and incompressible solid
TODO: Not yet fully implemented!
"""

import ambit_fe

import sys, traceback
import numpy as np
from pathlib import Path
import pytest


@pytest.mark.fsi
@pytest.mark.fluid_solid
@pytest.mark.skip(reason="Not yet ready for testing.")
def main():

    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS            = {'problem_type'          : 'fsi',
                            'USE_MIXED_DOLFINX_BRANCH' : True,
                            'write_results_every'   : 1,
                            'output_path'           : basepath+'/tmp/',
                            'mesh_domain'           : basepath+'/input/artseg-fsi-hex-quad_domain.xdmf',
                            'mesh_boundary'         : basepath+'/input/artseg-fsi-hex-quad_boundary.xdmf',
                            'results_to_write'      : [['displacement'], [['fluiddisplacement','velocity','pressure'],['aledisplacement','alevelocity']]],
                            'domain_ids_solid'      : [1], 
                            'domain_ids_fluid'      : [2],
                            'surface_ids_interface' : [1],
                            'simname'               : 'fsi_taylorhood_artseg',
                            }

    SOLVER_PARAMS        = {'solve_type'            : 'direct',
                            'direct_solver'         : 'mumps',
                            'tol_res'               : [1.0e-6,1.0e-8,1.0e-8,1.0e-8,1.0e-8,1.0e-1],
                            'tol_inc'               : [1.0e-1,1.0e-3,1.0e-1,1.0e-3,1.0e-3,1.0e-1]}

    TIME_PARAMS_SOLID    = {'maxtime'               : 1.0,
                            'numstep'               : 100,
                            #'numstep_stop'          : 1,
                            'timint'                : 'ost',
                            'theta_ost'             : 1.0}

    TIME_PARAMS_FLUID    = {'maxtime'               : 1.0,
                            'numstep'               : 100,
                            #'numstep_stop'          : 1,
                            'timint'                : 'ost',
                            'theta_ost'             : 1.0}

    FEM_PARAMS_SOLID     = {'order_disp'            : 2,
                            'order_pres'            : 1,
                            'quad_degree'           : 5,
                            'incompressible_2field' : True}

    FEM_PARAMS_FLUID     = {'order_vel'             : 2,
                            'order_pres'            : 1,
                            'quad_degree'           : 5}
    
    FEM_PARAMS_ALE       = {'order_disp'            : 2,
                            'quad_degree'           : 5}
    
    COUPLING_PARAMS      = {'coupling_fluid_ale'    : [{'surface_ids' : [1], 'type' : 'strong_dirichlet'}],
                            'fsi_governing_type'    : 'solid_governed', # solid_governed, fluid_governed
                            'fluid_on_deformed'     : 'consistent'}

    MATERIALS_SOLID      = {'MAT1' : {'neohooke_dev'      : {'mu' : 100.},
                                      #'sussmanbathe_vol'  : {'kappa' :500.},
                                      'inertia'           : {'rho0' : 1.0e-6},
                                      'visco_green'       : {'eta' : 0.1}}}

    MATERIALS_FLUID      = {'MAT1' : {'newtonian' : {'mu' : 4.0e-6},
                                      'inertia' : {'rho' : 1.025e-6}}}
    
    MATERIALS_ALE        = {'MAT1' : {'linelast' : {'Emod' : 10.0, 'kappa' : 100.}}}


    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    # some examples... up to 9 possible (tc1 until tc9 - feel free to implement more in timeintegration.py --> timecurves function if needed...)
    class time_curves():

        def tc1(self, t):
            t_ramp = 2.0
            p0 = 0.0
            pinfl = 0#0.1
            return (0.5*(-(pinfl-p0))*(1.-np.cos(np.pi*t/t_ramp)) + (-p0)) * (t<t_ramp) + (-pinfl)*(t>=t_ramp)


    BC_DICT_SOLID        = { 'dirichlet' : [{'id' : [2,3], 'dir' : 'z', 'val' : 0.},
                                            {'id' : [4], 'dir' : 'y', 'val' : 0.},
                                            {'id' : [5], 'dir' : 'x', 'val' : 0.}] }

    BC_DICT_FLUID        = { 'neumann' :   [{'id' : [2,3], 'dir' : 'normal_cur', 'curve' : 1}],
                             'dirichlet' : [{'id' : [4], 'dir' : 'y', 'val' : 0.},
                                            {'id' : [5], 'dir' : 'x', 'val' : 0.}] }

    BC_DICT_ALE          = { 'dirichlet' : [{'id' : [2,3], 'dir' : 'z', 'val' : 0.},
                                            {'id' : [4], 'dir' : 'y', 'val' : 0.},
                                            {'id' : [5], 'dir' : 'x', 'val' : 0.}] }



    # problem setup
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, [TIME_PARAMS_SOLID, TIME_PARAMS_FLUID], SOLVER_PARAMS, [FEM_PARAMS_SOLID, FEM_PARAMS_FLUID, FEM_PARAMS_ALE], [MATERIALS_SOLID, MATERIALS_FLUID, MATERIALS_ALE], [BC_DICT_SOLID, BC_DICT_FLUID, BC_DICT_ALE], time_curves=time_curves(), coupling_params=COUPLING_PARAMS)

    # problem solve
    problem.solve_problem()

    ## --- results check
    #tol = 1.0e-6

    #s_corr = np.zeros(problem.mp.pbf.cardvasc0D.numdof)


    #check1 = ambit_fe.resultcheck.results_check_vec(problem.mp.pbf.s, s_corr, problem.mp.comm, tol=tol)
    #success = ambit_fe.resultcheck.success_check([check1], problem.mp.comm)

    #return success

    #if not success:
        #raise RuntimeError("Test failed!")



if __name__ == "__main__":

    test_main()
