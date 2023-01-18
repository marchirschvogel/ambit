#!/usr/bin/env python3

import ambit

import sys
import numpy as np
from pathlib import Path

def main():
    
    basepath = str(Path(__file__).parent.absolute())

    # all possible input parameters

    IO_PARAMS            = {'problem_type'          : 'solid_flow0d', # solid, fluid, flow0d, solid_flow0d, fluid_flow0d, solid_flow0d_multiscale_gandr, solid_constraint
                            'use_model_order_red'   : False, # Model Order Reduction via Proper Orthogonal Decomposition (POD), for solid or fluid mechanics and 3D0D coupled problems (default: False); specify parameters in ROM_PARAMS (see below)
                            'mesh_domain'           : basepath+'/input/blocks_domain.xdmf', # domain mesh file
                            'mesh_boundary'         : basepath+'/input/blocks_boundary.xdmf', # boundary mesh file
                            'meshfile_type'         : 'ASCII', # OPTIONAL: type of encoding of your mesh file (ASCII or HDF5) (default: 'ASCII')
                            'fiber_data'            : {'nodal' : [basepath+'/file1.txt',basepath+'/file2.txt'], 'readin_tol' : 1.0e-8}, # OPTIONAL: only for anisotropic solid materials - nodal: fiber input data is stored at node coordinates, elemental: fiber input data is stored at element center; readin_tol - tolerance for readin (OPTIONAL: default is 1.0e-8, but depending on units, should be adapted)
                            'write_results_every'   : 1, # frequency for results output (negative value for no output, 1 for every time step, etc.)
                            'write_results_every_0D': 1, # OPTIONAL: for flow0d results (default: write_results_every)
                            'write_restart_every'   : 1, # OPTIONAL: if restart info should be written (default: -1)
                            'output_path'           : basepath+'/tmp/', # where results are written to
                            'output_path_0D'        : basepath+'/tmp/', # OPTIONAL: different output path for flow0d results (default: output_path)
                            'results_to_write'      : ['displacement','velocity','pressure','cauchystress'], # see io_routines.py for what to write
                            'simname'               : 'my_simulation_name', # how to name the output (attention: there is no warning, results will be overwritten if existent)
                            'restart_step'          : 0} # OPTIONAL: at which time step to restart a former simulation (that crashed and shoud be resumed or whatever) (default: 0)

    # for solid*, fluid* problem types
    SOLVER_PARAMS_SOLID  = {'solve_type'            : 'direct', # direct, iterative
                            'tol_res'               : 1.0e-8, # residual tolerance for nonlinear solver
                            'tol_inc'               : 1.0e-8, # increment tolerance for nonlinear solver
                            'divergence_continue'   : None, # OPTIONAL: what to apply when Newton diverges: None, 'PTC' ('ptc' can stay False) (default: None)
                            'ptc'                   : False, # OPTIONAL: if you want to use PTC straight away (independent of divergence_continue) (default: False)
                            'k_ptc_initial'         : 0.1, # OPTIONAL: initial PTC value that adapts during nonlinear iteration (default: 0.1)
                            'ptc_randadapt_range'   : [0.85, 1.35], # OPTIONAL: in what range to randomly adapt PTC parameter if divergence continues to occur (default: [0.85, 1.35]) (only if divergence_continue is set to 'PTC')
                            # direct linear solver settings (only apply for solve_type 'direct')
                            'direct_solver'         : 'superlu_dist', # OPTIONAL: type of direct solver: 'superlu_dist' or 'mumps' (default: 'superlu_dist')
                            # iterative linear solver settings (only apply for solve_type 'iterative')
                            'tol_lin'               : 1.0e-6, # OPTIONAL: linear solver tolerance (default: 1.0e-8)
                            'max_liniter'           : 1200, # OPTIONAL: maximum number of linear iterations (default: 1200)
                            'print_liniter_every'   : 50, # OPTIONAL: how often to print linear iterations (default: 50)
                            'adapt_linsolv_tol'     : False, # OPTIONAL: True, False - adapt linear tolerance throughout nonlinear iterations (default: False)
                            'adapt_factor'          : 0.1, # OPTIONAL: adaptation factor for adapt_linsolv_tol (the larger, the more adaptation) (default: 0.1)
                            # for local Newton (only for inelastic nonlinear materials at Gauss points, i.e. deformation-dependent growth)
                            'print_local_iter'      : False, # OPTIONAL: if we want to print iterations of local Newton (default: False)
                            'tol_res_local'         : 1.0e-10, # OPTIONAL: local Newton residual inf-norm tolerance (default: 1.0e-10)
                            'tol_inc_local'         : 1.0e-10} # OPTIONAL: local Newton increment inf-norm tolerance (default: 1.0e-10)
    
    # for flow0d, solid_flow0d, or fluid_flow0d problem types
    SOLVER_PARAMS_FLOW0D = {'tol_res'               : 1.0e-6, # residual tolerance for nonlinear solver
                            'tol_inc'               : 1.0e-6} # increment tolerance for nonlinear solver

    # for solid*, fluid* problem types
    TIME_PARAMS_SOLID    = {'maxtime'               : 1.0, # maximum simulation time
                            'numstep'               : 500, # number of steps over maxtime (maxtime/numstep governs the time step size)
                            'numstep_stop'          : 5, # OPTIONAL: if we want the simulation to stop earlier (default: numstep)
                            'timint'                : 'genalpha', # time-integration algorithm: genalpha, ost, static
                            'theta_ost'             : 1.0, # One-Step-Theta (ost) time integration factor 
                            'rho_inf_genalpha'      : 0.8} # spectral radius of Generalized-alpha (genalpha) time-integration (governs all other parameters alpha_m, alpha_f, beta, gamma)
    
    # for flow0d, solid_flow0d, or fluid_flow0d problem types
    TIME_PARAMS_FLOW0D   = {'timint'                : 'ost', # time-integration algorithm: ost
                            'theta_ost'             : 0.5, # One-Step-Theta time integration factor 
                            'initial_conditions'    : init(), # initial condition dictionary (here defined as function, see below)
                            'initial_file'          : None, # OPTIONAL: if we want to read initial conditions from a file (overwrites above specified dict)
                            'eps_periodic'          : 1.0e-3, # OPTIONAL: cardiac cycle periodicity tolerance (default: 1.0e-20)
                            'periodic_checktype'    : None, # OPTIONAL: None, ['allvar'], ['allvaraux'], ['pQvar'], ['specific',['V_v_l','p_ar_sys',...,'q_ven_pul']] (default: None)
                            'initial_backwardeuler' : False} # OPTIONAL: do Backward Euler for initial time step regardless of value for theta_ost (default: False)

    # for flow0d, solid_flow0d, or fluid_flow0d problem types
    MODEL_PARAMS_FLOW0D  = {'modeltype'             : 'syspul', # '2elwindkessel', '4elwindkesselLsZ', '4elwindkesselLpZ', 'syspul', 'syspulcap', 'syspulcaprespir'
                            'coronary_model'        : None, # OPTIONAL: coronary submodel - None, 'ZCRp_CRd', 'ZCRp_CRd_lr' (default: None)
                            'parameters'            : param(), # parameter dictionary (here defined as function, see below)
                            'chamber_models'        : {'lv' : {'type' : '3D_solid'}, 'rv' : {'type' : '3D_fluid', 'num_inflows' : 1, , 'num_outflows' : 1}, 'la' : {'type' : '0D_elast', 'activation_curve' : 5}, 'ra' : {'type' : '0D_prescr', 'prescribed_curve' : 5}}, # only for syspul* models - 3D_solid, 3D_fluid: chamber is 3D solid or fluid mechanics model, 0D_elast: chamber is 0D elastance model, 0D_prescr: volume/flux is prescribed over time, prescr_elast: chamber is 0D elastance model with prescribed elastance over time
                            'prescribed_variables'  : {'q_vin_l' : 1}, # OPTIONAL: in case we want to prescribe values: variable name, and time curve number (define below)
                            'perturb_type'          : None, # OPTIONAL: ['mr',1.0e-6], ['ms',25.], ['ar',5.0e-6], ['as',50.], ['mi',0.,4] (default: None)
                            'perturb_after_cylce'   : 1, # OPTIONAL: after which cycle to induce the perturbation / disease / cardiovascular state change... (default: -1)
                            'valvelaws'             : {'av' : ['pwlin_pres',0], 'mv' : ['pwlin_pres',0], 'pv' : ['pwlin_pres',0], 'tv' : ['pwlin_pres',0]}} # OPTIONAL: valve laws for aortic (av), mitral (mv), pulmonary (pv), and tricuspid valve (tv) (pwlin_pres: piecewise-linear pressure-governed, pwlin_time: piecewise-linear time-governed, smooth_pres_momentum: , smooth p-q relationship, smooth_pres_resistance: smooth resistance pressure-governed with number being amount of smoothness) (default: {'av' : ['pwlin_pres',0], 'mv' : ['pwlin_pres',0], 'pv' : ['pwlin_pres',0], 'tv' : ['pwlin_pres',0]})

    # for solid*, fluid* problem types
    FEM_PARAMS           = {'order_disp'            : 1, # order of displacement interpolation (solid mechanics)
                            'order_vel'             : 1, # order of velocity interpolation (fluid mechanics)
                            'order_pres'            : 1, # order of pressure interpolation (solid, fluid mechanics)
                            'quad_degree'           : 1, # quadrature degree q (number of integration points: n(q) = ((q+2)//2)**dim) --> can be 1 for linear tets, should be >= 3 for linear hexes, should be >= 5 for quadratic tets/hexes
                            'incompressible_2field' : False, # if we want to use a 2-field functional for pressure dofs (always applies for fluid, optional for solid mechanics)
                            'prestress_initial'     : False} # OPTIONAL: if we want to use MULF prestressing (Gee et al. 2010) prior to solving a dynamic/other kind of solid or solid-coupled problem (experimental, not thoroughly tested!) (default: False)
    
    # for solid_flow0d or fluid_flow0d problem type
    COUPLING_PARAMS      = {'surface_ids'           : [[1],[2]], # coupling surfaces (for syspul* models: order is lv, rv, la, ra - has to be consistent with chamber_models dict)
                            'surface_p_ids'         : [[1],[2]], # OPTIONAL: if pressure should be applied to different surface than that from which the volume/flux is measured from... (default: surface_ids)
                            'cq_factor'             : [1.,1.], # OPTIONAL: if we want to scale the 3D volume or flux (e.g. for 2D solid models) (default: [1.] * number of surfaces)
                            'coupling_quantity'     : ['volume','volume'], # volume, flux, pressure (former two need 'monolithic_direct', latter needs 'monolithic_lagrange' as coupling_type)
                            'variable_quantity'     : ['pressure','pressure'], # OPTIONAL: pressure, flux, volume (former needs 'monolithic_direct', latter two need 'monolithic_lagrange' as coupling_type) (default: 'pressure')
                            'coupling_type'         : 'monolithic_direct', # monolithic_direct, monolithic_lagrange (ask MH for the difference... or try to find out in the code... :))
                            'eps_fd'                : 1e-6, # OPTIONAL: perturbation for monolithic_lagrange coupling (default: 1e-5)
                            'print_subiter'         : False, # OPTIONAL: print subiterations in case of monolithic_lagrange-type coupling (default: False)
                            'Nmax_periodicref'      : 10, # OPTIONAL: maximum heart cycles for solid_flow0d_periodicref problem (default: 10)
                            'restart_periodicref'   : 0} # OPTIONAL: restart outer cycle for solid_flow0d_periodicref problem (default: 0)

    # for solid_constraint problem type
    CONSTRAINT_PARAMS    = {'surface_ids'           : [[1],[2]], # coupling surfaces for volume or flux constraint (for syspul* models: order is lv, rv, la, ra)
                            'surface_p_ids'         : [[1],[2]], # OPTIONAL: if pressure should be applied to different surface than that from which the volume/flux is measured from... (default: surface_ids) (for syspul* models: order is lv, rv, la, ra)
                            'constraint_quantity'   : ['volume','volume'], # volume, flux, pressure (for syspul* models: order is lv, rv, la, ra) (default: volume) 
                            'prescribed_curve'      : [5,6]} # time curves that set the volumes/fluxes that shall be met

    # for model order reduction
    ROM_PARAMS           = {'hdmfilenames'          : [basepath+'/input/checkpoint_simname_u_*_1proc.dat'], # input files of high-dimensional model (HDM), need "*" indicating the numbered file series
                            'numhdms'               : 1, # number of high-dimensional models
                            'numsnapshots'          : 10, # number of snapshots
                            'snapshotincr'          : 1, # OPTIONAL: snapshot increment (default: 1)
                            'numredbasisvec'        : 10, # OPTIONAL: number of reduced basis vectors to consider (default: numsnapshots)
                            'eigenvalue_cutoff'     : 1.0e-8, # OPTIONAL: cutoff tolerance (discard eigenvalues lower than that) (default: 0.0)
                            'print_eigenproblem'    : False, # OPTIONAL: print output of Proper Orthogonal Decomposition (POD) eigensolve (default: False)
                            'surface_rom'           : [1], # OPTIONAL: apply reduced-order model only to a (set of) surface(s) specified by boundary id(s) (default: [])
                            'snapshotsource'        : 'petscvector', # OPTIONAL: source of snapshot data: 'petscvector' or 'rawtxt' (default: 'petscvector')
                            'write_pod_modes'       : False} # OPTIONAL: whether to write out POD modes (default: False)

    # for solid_flow0d_multiscale_gandr problem type
    MULTISCALE_GR_PARAMS = {'gandr_trigger_phase'   : 'end_diastole', # end_diastole, end_systole
                            'numcycles'             : 10, # max. number of multiscale cycles (one cycle means one small scale succeeded by a large scale run)
                            'tol_small'             : 1.0e-3, # cycle error tolerance: overrides eps_periodic from TIME_PARAMS_FLOW0D
                            'tol_large'             : 1.0e-4, # growth rate tolerance
                            'tol_outer'             : 1.0e-3, # tolerance for volume increase during one growth cycle - stop sim if equal to or below this value
                            'write_checkpoints'     : False, # OPTIONAL: to write checkpoints after each small or large scale run to restart from there (default: False)
                            'restart_cycle'         : 0, # OPTIONAL: at which multiscale cycle to restart (default: 0)
                            'restart_from_small'    : False} # OPTIONAL: if the multiscale sim should be restarted from a previous small scale run (small scale of restart_cycle needs to be computed already) (default: False)
                            
                            # - MATn has to correspond to subdomain id n (set by the flags in Attribute section of *_domain.xdmf file - so if you have x mats, you need ids ranging from 1,...,x)
                            # - one MAT can be decomposed into submats, see examples below (additive stress contributions)
                            # - for solid: if you use a deviatoric (_dev) mat, you should also use EITHER a volumetric (_vol) mat, too, OR set incompressible_2field in FEM_PARAMS to 'True' and then only use a _dev mat and MUST NOT use a _vol mat! (if incompressible_2field is 'True', then all materials have to be treated perfectly incompressible currently)
                            # - for fluid: incompressible_2field is always on, and you only can define a Newtonian fluid ('newtonian') with dynamic viscosity 'eta'
                            # - for dynamics, you need to specify a mat called 'inertia' and set the density ('rho0' in solid, 'rho' in fluid dynamics)
                            # - material can also be inelastic and growth ('growth')
                            # - see solid_material.py or fluid_material.py for material laws available (and their parameters), and feel free to implement/add new strain energy functions or laws fairly quickly
    MATERIALS            = {'MAT1' : {'holzapfelogden_dev' : {'a_0' : 0.059, 'b_0' : 8.023, 'a_f' : 18.472, 'b_f' : 16.026, 'a_s' : 2.481, 'b_s' : 11.120, 'a_fs' : 0.216, 'b_fs' : 11.436, 'fiber_comp' : False},
                                      'sussmanbathe_vol'   : {'kappa' : 1.0e3},
                                      'active_fiber'       : {'sigma0' : 50.0, 'alpha_max' : 15.0, 'alpha_min' : -20.0, 'activation_curve' : 4, 'frankstarling' : True, 'amp_min' : 1., 'amp_max' : 1.7, 'lam_threslo' : 1.01, 'lam_maxlo' : 1.15, 'lam_threshi' : 999., 'lam_maxhi' : 9999.},
                                      'inertia'            : {'rho0' : 1.0e-6},
                                      'rayleigh_damping'   : {'eta_m' : 0.0, 'eta_k' : 0.0001}},
                            'MAT2' : {'neohooke_dev'       : {'mu' : 10.},
                                      'ogden_vol'          : {'kappa' : 10./(1.-2.*0.49)},
                                      'inertia'            : {'rho0' : 1.0e-6},
                                      'growth'             : {'growth_dir' : 'isotropic', # isotropic, fiber, crossfiber, radial
                                                              'growth_trig' : 'volstress', # fibstretch, volstress, prescribed
                                                              'growth_thres' : 1.01, # critial value above which growth happens (i.e. a critial stretch, stress or whatever depending on the growth trigger)
                                                              'thres_tol' : 1.0e-4, # tolerance for threshold (makes sense in multiscale approach, where threshold is set element-wise)
                                                              'trigger_reduction' : 1, # reduction factor for trigger ]0,1]
                                                              'thetamax' : 1.5, # maximum growth stretch
                                                              'thetamin' : 1.0, # minimum growth stretch
                                                              'tau_gr' : 1.0, # growth time constant
                                                              'gamma_gr' : 2.0, # growth nonlinearity
                                                              'tau_gr_rev' : 1000.0, # reverse growth time constant
                                                              'gamma_gr_rev' : 2.0, # reverse growth nonlinearity
                                                              'remodeling_mat' : {'neohooke_dev' : {'mu' : 3.}, # remodeling material
                                                                                  'ogden_vol'    : {'kappa' : 3./(1.-2.*0.49)}}}}}


    # define your load curves here (syntax: tcX refers to curve X, to be used in BC_DICT key 'curve' : [X,0,0], or 'curve' : X)
    # some examples... up to 9 possible (tc1 until tc9 - feel free to implement more in timeintegration.py --> timecurves function if needed...)
    class time_curves():
        
        def tc1(self, t):
            return 3.*t
        
        def tc2(self, t):
            return -5000.0*np.sin(2.*np.pi*t/TIME_PARAMS_SOLID['maxtime'])

        def tc3(self, t): # can be a constant but formally needs t as input
            return 5.

        def tc4(self, t): # for active stress activation
            
            K = 5.
            t_contr, t_relax = 0.2, 0.53
            
            alpha_max = MATERIALS['MAT1']['active_fiber']['alpha_max']
            alpha_min = MATERIALS['MAT1']['active_fiber']['alpha_min']
            
            c1 = t_contr + alpha_max/(K*(alpha_max-alpha_min))
            c2 = t_relax - alpha_max/(K*(alpha_max-alpha_min))
            
            # Diss Hirschvogel eq. 2.101
            return (K*(t-c1)+1.)*((K*(t-c1)+1.)>0.) - K*(t-c1)*((K*(t-c1))>0.) - K*(t-c2)*((K*(t-c2))>0.) + (K*(t-c2)-1.)*((K*(t-c2)-1.)>0.)

        def tc5(self, t): # 0D elastance activation function
            
            act_dur = 0.4
            t0 = 0.
            
            if t >= t0 and t <= t0 + act_dur:
                y = 0.5*(1.-np.cos(2.*np.pi*(t-t0)/act_dur))
            else:
                y = 0.0

        #...

    # bc syntax examples
    BC_DICT              = { 'dirichlet' : [{'id' : [1], 'dir' : 'all', 'val' : 0.}, # either curve or val
                                            {'id' : [2,4,5], 'dir' : 'y', 'val' : 0.}, # either curve or val
                                            {'id' : [3], 'dir' : 'z', 'curve' : 1}], # either curve or val
                            # Neumann can be - pk1 with dir xyz (then use 'curve' : [xcurve-num, ycurve-num, zcurve-num] with 0 meaning zero),
                            #                - pk1 with dir normal (then use 'curve' : curve-num with 0 meaning zero)
                            #                - true with dir normal (then use 'curve' : curve-num with 0 meaning zero) = follower load in current normal direction
                            'neumann'    : [{'type' : 'pk1', 'id' : [3], 'dir' : 'xyz', 'curve' : [1,0,0]},
                                            {'type' : 'pk1', 'id' : [2], 'dir' : 'normal', 'curve' : 1},
                                            {'type' : 'true', 'id' : [2], 'dir' : 'normal', 'curve' : 1}],
                            # Robib BC can be either spring or dashpot, both either in xyz or normal direction
                            'robin'      : [{'type' : 'spring', 'id' : [3], 'dir' : 'normal', 'stiff' : 0.075},
                                            {'type' : 'dashpot', 'id' : [3], 'dir' : 'xyz', 'visc' : 0.005}] }

    # problem setup - exemplary for 3D-0D coupling of solid (fluid) to flow0d
    problem = ambit.Ambit(IO_PARAMS, [TIME_PARAMS_SOLID, TIME_PARAMS_FLOW0D], [SOLVER_PARAMS_SOLID, SOLVER_PARAMS_FLOW0D], FEM_PARAMS, [MATERIALS, MODEL_PARAMS_FLOW0D], BC_DICT, time_curves=time_curves(), coupling_params=COUPLING_PARAMS, multiscale_params=MULTISCALE_GR_PARAMS, mor_params=ROM_PARAMS)
    
    # problem setup for solid (fluid) only: just pass parameters related to solid (fluid) instead of lists, so:
    #problem = ambit.Ambit(IO_PARAMS, TIME_PARAMS_SOLID, SOLVER_PARAMS_SOLID, FEM_PARAMS, MATERIALS, BC_DICT, time_curves=time_curves(), mor_params=ROM_PARAMS)

    # problem solve
    problem.solve_problem()


# syspul circulation model initial condition and parameter dicts...

def init():
    
    return {'q_vin_l_0' : 1.1549454594333263E+04,
            'p_at_l_0' : 3.8580961077622145E-01,
            'q_vout_l_0' : -1.0552685263595845E+00,
            'p_v_l_0' : 3.7426015618188813E-01,
            'p_ar_sys_0' : 1.0926945419777734E+01,
            'q_ar_sys_0' : 7.2237210814547114E+04,
            'p_ven_sys_0' : 2.2875736545217800E+00,
            'q_ven_sys_0' : 8.5022643486798144E+04,
            'q_vin_r_0' : 4.1097788677528049E+04,
            'p_at_r_0' : 2.4703021083862464E-01,
            'q_vout_r_0' : -2.0242075369768467E-01,
            'p_v_r_0' : 2.0593242216109664E-01,
            'p_ar_pul_0' : 2.2301399591379436E+00,
            'q_ar_pul_0' : 3.6242987765574515E+04,
            'p_ven_pul_0' : 1.6864951426543255E+00,
            'q_ven_pul_0' : 8.6712368791873596E+04,
            # coronary circulation submodel
            'q_cor_sys_0' : 0,
            'q_cord_sys_0' : 0,
            'q_corp_sys_0' : 0.,
            'p_cord_sys_0' : 0.}

def param():
    
    R_ar_sys = 120.0e-6
    tau_ar_sys = 1.65242332
    tau_ar_pul = 0.3

    # Diss Hirschvogel tab. 2.7
    C_ar_sys = tau_ar_sys/R_ar_sys
    Z_ar_sys = R_ar_sys/20.
    R_ven_sys = R_ar_sys/5.
    C_ven_sys = 30.*C_ar_sys
    R_ar_pul = R_ar_sys/8.
    C_ar_pul = tau_ar_pul/R_ar_pul
    Z_ar_pul = 0.
    R_ven_pul = R_ar_pul
    C_ven_pul = 2.5*C_ar_pul

    L_ar_sys = 0.667e-6
    L_ven_sys = 0.
    L_ar_pul = 0.
    L_ven_pul = 0.

    # atrial elastances
    E_at_A_l, E_at_min_l = 20.0e-6, 9.0e-6
    E_at_A_r, E_at_min_r = 10.0e-6, 8.0e-6

    # timings
    t_ed = 0.2
    t_es = 0.53


    return {'R_ar_sys' : R_ar_sys,
            'C_ar_sys' : C_ar_sys,
            'L_ar_sys' : L_ar_sys,
            'Z_ar_sys' : Z_ar_sys,
            'R_ar_pul' : R_ar_pul,
            'C_ar_pul' : C_ar_pul,
            'L_ar_pul' : L_ar_pul,
            'Z_ar_pul' : Z_ar_pul,
            'R_ven_sys' : R_ven_sys,
            'C_ven_sys' : C_ven_sys,
            'L_ven_sys' : L_ven_sys,
            'R_ven_pul' : R_ven_pul,
            'C_ven_pul' : C_ven_pul,
            'L_ven_pul' : L_ven_pul,
            # atrial elastances
            'E_at_max_l' : E_at_min_l+E_at_A_l,
            'E_at_min_l' : E_at_min_l,
            'E_at_max_r' : E_at_min_r+E_at_A_r,
            'E_at_min_r' : E_at_min_r,
            # ventricular elastances
            'E_v_max_l' : 7.0e-5,
            'E_v_min_l' : 12.0e-6,
            'E_v_max_r' : 3.0e-5,
            'E_v_min_r' : 10.0e-6,
            # valve resistances
            'R_vin_l_min' : 1.0e-6,
            'R_vin_l_max' : 1.0e1,
            'R_vout_l_min' : 1.0e-6,
            'R_vout_l_max' : 1.0e1,
            'R_vin_r_min' : 1.0e-6,
            'R_vin_r_max' : 1.0e1,
            'R_vout_r_min' : 1.0e-6,
            'R_vout_r_max' : 1.0e1,
            # valve inertances
            'L_vin_l' : 0,
            'L_vin_r' : 0,
            'L_vout_l' : 0,
            'L_vout_r' : 0,
            # timings
            't_ed' : t_ed,
            't_es' : t_es,
            'T_cycl' : 1.0,
            # coronary circulation submodel parameters - values from Arthurs et al. 2016, Tab. 3
            'Z_corp_sys' : 3.2e-3,
            'C_corp_sys' : 4.5e0,
            'R_corp_sys' : 6.55e-3,
            'C_cord_sys' : 2.7e1,
            'R_cord_sys' : 1.45e-1
            # unstressed compartment volumes (for post-processing)
            'V_at_l_u' : 0.0,
            'V_at_r_u' : 0.0,
            'V_v_l_u' : 0.0,
            'V_v_r_u' : 0.0,
            'V_ar_sys_u' : 0.0,
            'V_ar_pul_u' : 0.0,
            'V_ven_sys_u' : 0.0,
            'V_ven_pul_u' : 0.0,
            # coronary circulation submodel
            'V_corp_sys_u' : 0.0,
            'V_cord_sys_u' : 0.0}



if __name__ == "__main__":
    
    main()
