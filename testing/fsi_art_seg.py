#!/usr/bin/env python3

### LV FSI

import ambit

import sys, traceback
import numpy as np
from pathlib import Path

import resultcheck

def main():

    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS            = {'problem_type'          : 'fsi',
                            'write_results_every'   : -999,
                            'output_path'           : basepath+'/tmp/',
                            'simname'               : 'test',
                            'io_solid' : {'mesh_domain'           : basepath+'/../../../sim/art_seg/solid_h1.0/h00q/__data__/input-quad_domain.xdmf',
                                          'mesh_boundary'         : basepath+'/../../../sim/art_seg/solid_h1.0/h00q/__data__/input-quad_boundary.xdmf',
                                          'results_to_write'      : [],
                                          'write_results_every'   : -999,
                                          'output_path'           : basepath+'/tmp/'},
                            'io_fluid' : {'mesh_domain'           : basepath+'/../../../sim/art_seg/frsi/h00q/__data__/input-quad_domain.xdmf',
                                          'mesh_boundary'         : basepath+'/../../../sim/art_seg/frsi/h00q/__data__/input-quad_boundary.xdmf',
                                          'results_to_write'      : [[],[]],
                                          'write_results_every'   : -999,
                                          'output_path'           : basepath+'/tmp/'}
                            }

    SOLVER_PARAMS        = {'solve_type'            : 'direct',
                            'tol_res'               : 1.0e-8,
                            'tol_inc'               : 1.0e-8}

    TIME_PARAMS_SOLID    = {'maxtime'               : 1.0,
                            'numstep'               : 100,
                            'numstep_stop'          : 1,
                            'timint'                : 'ost',
                            'theta_ost'             : 1.0}

    TIME_PARAMS_FLUID    = {'maxtime'               : 1.0,
                            'numstep'               : 100,
                            'numstep_stop'          : 1,
                            'timint'                : 'ost',
                            'theta_ost'             : 1.0}

    FEM_PARAMS_SOLID     = {'order_disp'            : 2,
                            'order_pres'            : 1,
                            'quad_degree'           : 5,
                            'incompressible_2field' : True}

    FEM_PARAMS_FLUID     = {'order_vel'             : 2,
                            'order_pres'            : 1,
                            'quad_degree'           : 5}
    
    COUPLING_PARAMS      = {'coupling_fluid_ale'    : {'surface_ids' : [1], 'type' : 'strong_dirichlet'},
                            'fluid_solid_interface' : 'solid_governed', # solid_governed, fluid_governed
                            'fluid_on_deformed'     : 'consistent'}

    MATERIALS_SOLID      = {'MAT1' : {'neohooke_dev'      : {'mu' : 10.},
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
            pinfl = 5.0
            return (0.5*(-(pinfl-p0))*(1.-np.cos(np.pi*t/t_ramp)) + (-p0)) * (t<t_ramp) + (-pinfl)*(t>=t_ramp)


    BC_DICT_SOLID        = { 'dirichlet' : [{'id' : [2,3], 'dir' : 'all', 'val' : 0.},
                                            {'id' : [4], 'dir' : 'y', 'val' : 0.},
                                            {'id' : [5], 'dir' : 'x', 'val' : 0.}] }

    BC_DICT_FLUID        = { 'neumann' :   [{'id' : [2,3], 'dir' : 'normal_cur', 'curve' : 1}],
                             'dirichlet' : [{'id' : [4], 'dir' : 'y', 'val' : 0.},
                                            {'id' : [5], 'dir' : 'x', 'val' : 0.}] }

    BC_DICT_ALE          = { 'dirichlet' : [{'id' : [2,3], 'dir' : 'z', 'val' : 0.},
                                            {'id' : [4], 'dir' : 'y', 'val' : 0.},
                                            {'id' : [5], 'dir' : 'x', 'val' : 0.}] }

    # problem setup
    problem = ambit.Ambit(IO_PARAMS, [TIME_PARAMS_SOLID, TIME_PARAMS_FLUID], SOLVER_PARAMS, [FEM_PARAMS_SOLID, FEM_PARAMS_FLUID], [MATERIALS_SOLID, MATERIALS_FLUID, MATERIALS_ALE], [BC_DICT_SOLID, BC_DICT_FLUID, BC_DICT_ALE], time_curves=time_curves(), coupling_params=COUPLING_PARAMS)

    # problem solve
    problem.solve_problem()

    ## --- results check
    #tol = 1.0e-6

    #s_corr = np.zeros(problem.mp.pbf.cardvasc0D.numdof)


    #check1 = resultcheck.results_check_vec(problem.mp.pbf.s, s_corr, problem.mp.comm, tol=tol)
    #success = resultcheck.success_check([check1], problem.mp.comm)

    #return success






if __name__ == "__main__":

    success = False

    try:
        success = main()
    except:
        print(traceback.format_exc())

    if success:
        sys.exit(0)
    else:
        sys.exit(1)
