#!/usr/bin/env python3

### LV FSI

import ambit

import sys, traceback
import numpy as np
from pathlib import Path

import resultcheck

def main():
    
    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS            = {'problem_type'          : 'fluid_ale',
                            'write_results_every'   : -999,
                            'output_path'           : basepath+'/tmp/',
                            'mesh_domain'           : basepath+'/../../../sim/lv2/fsi/h01q/__data__/input-quad-fluid_domain.xdmf',
                            'mesh_boundary'         : basepath+'/../../../sim/lv2/fsi/h01q/__data__/input-quad-fluid_boundary.xdmf',
                            'results_to_write'      : [],
                            'simname'               : 'fluid_ale'}

    SOLVER_PARAMS        = {'solve_type'            : 'direct',
                            'tol_res'               : 1.0e-8,
                            'tol_inc'               : 1.0e-8}

    TIME_PARAMS          = {'maxtime'               : 1.0,
                            'numstep'               : 100,
                            'numstep_stop'          : 1,
                            'timint'                : 'ost',
                            'theta_ost'             : 1.0}

    FEM_PARAMS           = {'order_vel'             : 2, 
                            'order_pres'            : 1,
                            'quad_degree'           : 5}
    
    COUPLING_PARAMS      = {'surface_ids'           : [1]}

    MATERIALS_FLUID      = { 'MAT1' : {'newtonian' : {'eta' : 4.0e-6},
                                      'inertia' : {'rho' : 1.025e-6}} }

    MATERIALS_ALE        = { 'MAT1' : {'helmholtz' : {'k' : 1.0}} }


    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    # some examples... up to 9 possible (tc1 until tc9 - feel free to implement more in timeintegration.py --> timecurves function if needed...)
    class time_curves():
        
        def tc1(self, t):
            pmax = 1.0
            return -pmax*t/TIME_PARAMS['maxtime']


    BC_DICT_ALE          = { 'dirichlet' : [{'id' : [2], 'dir' : 'all', 'val' : 0.}] } # in-/outflow

    BC_DICT_FLUID        = { 'membrane' : [{'id' : [1], 'params' : {'model' : 'membrane_f', 'a_0' : 1.0, 'b_0' : 6.0, 'eta' : 0.2, 'rho0' : 1e-4, 'h0' : 10.}}],
                             'neumann' : [{'type' : 'true', 'id' : [2], 'dir' : 'normal', 'curve' : 1}] } # in-/outflow

    # problem setup
    problem = ambit.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, FEM_PARAMS, [MATERIALS_FLUID, MATERIALS_ALE], [BC_DICT_FLUID, BC_DICT_ALE], time_curves=time_curves(), coupling_params=COUPLING_PARAMS)
    
    # problem solve
    problem.solve_problem()

    ## --- results check
    #tol = 1.0e-6
        
    #s_corr = np.zeros(problem.mp.pbf.cardvasc0D.numdof)

    ## correct 0D results
    #s_corr[0] = 1.5140154884790708E+06
    #s_corr[1] = 1.2842189823315804E+00
    #s_corr[2] = -2.2969712709549372E+00
    #s_corr[3] = -2.2979650614749048E-01
    #s_corr[4] = 1.2035047941266040E+01
    #s_corr[5] = -2.2969712709549372E+00
    #s_corr[6] = 1.2035061723093666E+01
    #s_corr[7] = 7.9266376767811471E+03
    #s_corr[8] = 1.0920576465738236E+01
    #s_corr[9] = 4.8757248185760414E+04
    #s_corr[10] = 4.5874998121789955E+04
    #s_corr[11] = 3.5972406593841457E+04
    #s_corr[12] = 2.4558826877863074E+04
    #s_corr[13] = 8.1860969826577357E+03
    #s_corr[14] = 2.2677989771637215E+00
    #s_corr[15] = 8.3769512865623369E+02
    #s_corr[16] = 2.2702566758279015E+00
    #s_corr[17] = 9.5189348228574841E+02
    #s_corr[18] = 2.2702868360134905E+00
    #s_corr[19] = 7.5312469003723868E+02
    #s_corr[20] = 2.2701207039761671E+00
    #s_corr[21] = 5.0027875726087944E+02
    #s_corr[22] = 2.2705227157051970E+00
    #s_corr[23] = 1.7138027932289609E+02
    #s_corr[24] = 2.2541277926640513E+00
    #s_corr[25] = 2.0733859810437830E+05
    #s_corr[26] = 9.9018455409868766E+04
    #s_corr[27] = 2.7306272250688318E-01
    #s_corr[28] = -4.3686457082345953E-01
    #s_corr[29] = 1.7404426709701443E-01
    #s_corr[30] = 2.4017163229044409E+00
    #s_corr[31] = 1.1803825766722653E+04
    #s_corr[32] = 2.3131876296540206E+00
    #s_corr[33] = 2.0066547489885686E+05
    #s_corr[34] = 1.6159462113751601E+00
    #s_corr[35] = 3.9891468722433310E+04

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
