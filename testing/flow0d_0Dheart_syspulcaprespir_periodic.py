#!/usr/bin/env python3

import ambit

import sys, traceback
import numpy as np
from pathlib import Path

import results_check

# TODO: Pure computation time of the old testcase where we did not use Sympy and lambdify of expressions
# took only ~5.5 s of pure computing time
# now, we've slowed down to 107 s (so nearly 2 min) :-( --> We have to find better options than lambdify with numpy backend
# (ufuncify or theano_function could do the job...)

def main():
    
    basepath = str(Path(__file__).parent.absolute())

    IO_PARAMS         = {'problem_type'          : 'flow0d',
                         'write_results_every'   : -999,
                         'output_path'           : ''+basepath+'/tmp/',
                         'simname'               : 'test',
                         'prescribed_path'       : ''+str(basepath)+'/input'}

    SOLVER_PARAMS     = {'tol_res'               : 1.0e-7,
                         'tol_inc'               : 1.0e-7}

    TIME_PARAMS       = {'maxtime'               : 5.0,
                         'numstep'               : 500,
                         'timint'                : 'ost',
                         'theta_ost'             : 1.0,
                         'initial_conditions'    : init(),
                         'eps_periodic'          : 0.05,
                         'periodic_checktype'    : 'pvar'}
    
    MODEL_PARAMS      = {'modeltype'             : 'syspulcaprespir',
                         'parameters'            : param(),
                         'chamber_models'        : {'lv' : 'prescr_elast', 'rv' : 'prescr_elast', 'la' : 'prescr_elast', 'ra' : 'prescr_elast'},
                         'have_elastance'        : True}
    

    # define your time curves here (syntax: tcX refers to curve X)
    # None to be defined


    # problem setup
    problem = ambit.Ambit(IO_PARAMS, TIME_PARAMS, SOLVER_PARAMS, constitutive_params=MODEL_PARAMS)
    
    # solve time-dependent problem
    problem.solve_problem()


    # --- results check
    tol = 1.0e-6

    s_corr = np.zeros(problem.mp.cardvasc0D.numdof)

    ## correct results - from former baci testcase (5 cycles with each 1.0 s, theta = 1.0)
    ## currently CANNOT be reproduced with the prescribed elastance model here (even though we're not too far off...)
    ## TODO: Check why this is the case!
    #s_corr[0] = 7.71781462654294883e+04
    #s_corr[1] = 4.82455285374715770e-01
    #s_corr[2] = -9.32089188108091160e-01
    #s_corr[3] = 4.05277139109286311e-01
    #s_corr[4] = 9.72616902019019669e+00
    #s_corr[5] = 5.83726080022352398e+04
    #s_corr[6] = 9.32931790589786658e+00
    #s_corr[7] = 1.83084886403833043e+04
    #s_corr[8] = 1.72154627598280422e+04
    #s_corr[9] = 1.35034073594659403e+04
    #s_corr[10] = 9.21242124362669529e+03
    #s_corr[11] = 3.07080126056907102e+03
    #s_corr[12] = 2.41090621846982422e+00
    #s_corr[13] = 2.27052534769887789e+04
    #s_corr[14] = 2.41613895635923948e+00
    #s_corr[15] = 2.21779080874570500e+04
    #s_corr[16] = 2.41406095622704475e+00
    #s_corr[17] = 1.74172480658872846e+04
    #s_corr[18] = 2.41889648262861146e+00
    #s_corr[19] = 1.18412162444708483e+04
    #s_corr[20] = 2.41890958116365296e+00
    #s_corr[21] = 3.95711134217977497e+03
    #s_corr[22] = 2.04035648172536721e+00
    #s_corr[23] = 9.42308665820743918e+04
    #s_corr[24] = 7.15400334964196954e+04
    #s_corr[25] = 2.31123843349538866e-01
    #s_corr[26] = -2.39084978137291376e-01
    #s_corr[27] = 1.59583809853119180e-01
    #s_corr[28] = 2.55043359122603297e+00
    #s_corr[29] = 2.05678359828922803e+04
    #s_corr[30] = 2.39617482135434079e+00
    #s_corr[31] = 3.90560782940216595e+04
    #s_corr[32] = 2.10325423414917845e+00
    #s_corr[33] = 1.08053263251630851e+05
    #s_corr[34] = 4.77138986718484014e+06
    #s_corr[35] = 2.66700289957665256e+05
    #s_corr[36] = 9.99650347866085411e+01
    #s_corr[37] = 3.27022324023728411e-02
    #s_corr[38] = 1.48960517556904676e-01
    #s_corr[39] = 1.67086022107776043e+04
    #s_corr[40] = 1.66918635646860639e+04
    #s_corr[41] = 1.30961635410208564e+04
    #s_corr[42] = 8.89244395770556184e+03
    #s_corr[43] = 2.98353472804512467e+03
    #s_corr[44] = 1.59148286724528916e+01
    #s_corr[45] = 5.05719330574661186e+00
    #s_corr[46] = 1.59054445599070569e+01
    #s_corr[47] = 5.05719330574661186e+00
    #s_corr[48] = 1.58961192182105986e+01
    #s_corr[49] = 5.06015201758573507e+00
    #s_corr[50] = 3.46365073559847225e+00
    #s_corr[51] = 1.34592586031045460e+01
    #s_corr[52] = 4.11463113554487236e+00
    #s_corr[53] = 1.20061143012695695e+01
    #s_corr[54] = 4.12264630043372993e+00
    #s_corr[55] = 1.19870456277740534e+01
    #s_corr[56] = 4.12650550117093129e+00
    #s_corr[57] = 1.19774295132249193e+01
    #s_corr[58] = 4.13959751959834321e+00
    #s_corr[59] = 1.19526966777885217e+01
    #s_corr[60] = 8.15516536353003829e+00
    #s_corr[61] = 6.85478506275626120e+00
    #s_corr[62] = 4.42228745397937040e+00
    #s_corr[63] = 1.11168107086389227e+01
    #s_corr[64] = 4.87983353032786340e+01
    #s_corr[65] = 1.47121161937896550e+00
    #s_corr[66] = 7.33259028428000548e+00
    #s_corr[67] = 7.35917863339734968e+00
    #s_corr[68] = 6.40038489593068594e+00
    #s_corr[69] = 8.09752757832401571e+00
    #s_corr[70] = 8.15548041504590238e+00
    #s_corr[71] = 6.85197645062970562e+00
    #s_corr[72] = 4.42194570246950125e+00
    #s_corr[73] = 1.11159168023280195e+01
    #s_corr[74] = 4.87944514855986995e+01
    #s_corr[75] = 1.46148783974661067e+00
    #s_corr[76] = 7.33353820491393904e+00
    #s_corr[77] = 7.35448022540421764e+00
    #s_corr[78] = 6.40121826107596270e+00
    #s_corr[79] = 8.09305239721075331e+00
    #s_corr[80] = 1.59169639733867108e+01
    #s_corr[81] = 5.05686647907035258e+00
    
    # "new" "correct" results after 1 cycle
    s_corr[0] = 7.8772984062558084E+04
    s_corr[1] = 4.8659486302195032E-01
    s_corr[2] = -9.3610058358450243E-01
    s_corr[3] = 4.0782187895939220E-01
    s_corr[4] = 9.7688277148044182E+00
    s_corr[5] = 5.8808488437678505E+04
    s_corr[6] = 9.3690308416361692E+00
    s_corr[7] = 1.8467820872202938E+04
    s_corr[8] = 1.7334898156421506E+04
    s_corr[9] = 1.3597017271886743E+04
    s_corr[10] = 9.2764846741601486E+03
    s_corr[11] = 3.0920607948138941E+03
    s_corr[12] = 2.3904106904481224E+00
    s_corr[13] = 2.1963539583972666E+04
    s_corr[14] = 2.4078904587582968E+00
    s_corr[15] = 2.2186303961320122E+04
    s_corr[16] = 2.4058351324957048E+00
    s_corr[17] = 1.7424935125746462E+04
    s_corr[18] = 2.4105541578551586E+00
    s_corr[19] = 1.1842731275571505E+04
    s_corr[20] = 2.4107809114187750E+00
    s_corr[21] = 3.9598510095865377E+03
    s_corr[22] = 2.0319657244376885E+00
    s_corr[23] = 9.3711973674203065E+04
    s_corr[24] = 7.0539636117053349E+04
    s_corr[25] = 2.3269582989298973E-01
    s_corr[26] = -2.4254777064876870E-01
    s_corr[27] = 1.6215619377593637E-01
    s_corr[28] = 2.5876339002636235E+00
    s_corr[29] = 2.1024862315696508E+04
    s_corr[30] = 2.4299474328958999E+00
    s_corr[31] = 3.9854755419966961E+04
    s_corr[32] = 2.1310367672461474E+00
    s_corr[33] = 1.0962946028161315E+05
    s_corr[34] = 4.9359151070363959E+06
    s_corr[35] = 1.2176999298415647E+06
    s_corr[36] = 9.9835478479934423E+01
    s_corr[37] = 1.1874877807660576E-02
    s_corr[38] = 1.8905022661610985E-01
    s_corr[39] = 1.6856052226168646E+04
    s_corr[40] = 1.6807410235902895E+04
    s_corr[41] = 1.3186748889259841E+04
    s_corr[42] = 8.9541309449532910E+03
    s_corr[43] = 3.0041461413938432E+03
    s_corr[44] = 1.5868870030490044E+01
    s_corr[45] = 5.0645312522587602E+00
    s_corr[46] = 1.5853909749529379E+01
    s_corr[47] = 5.0669389175553903E+00
    s_corr[48] = 1.5849765885394156E+01
    s_corr[49] = 5.0676025646993645E+00
    s_corr[50] = 1.6195682913639453E+00
    s_corr[51] = 1.7179751665615537E+01
    s_corr[52] = 3.8185082152968373E+00
    s_corr[53] = 1.2477049172174398E+01
    s_corr[54] = 3.9607746097186181E+00
    s_corr[55] = 1.2246871673462810E+01
    s_corr[56] = 4.0976635194540458E+00
    s_corr[57] = 1.2021904244523247E+01
    s_corr[58] = 4.1363317713754935E+00
    s_corr[59] = 1.1959313574705780E+01
    s_corr[60] = 8.1560564804157494E+00
    s_corr[61] = 6.8543786891326901E+00
    s_corr[62] = 4.4220064405023898E+00
    s_corr[63] = 1.1117637256132248E+01
    s_corr[64] = 4.8799048063729643E+01
    s_corr[65] = 1.4670011536083674E+00
    s_corr[66] = 7.3334628304288074E+00
    s_corr[67] = 7.3581916462645314E+00
    s_corr[68] = 6.4011225251472208E+00
    s_corr[69] = 8.0967303185253847E+00
    s_corr[70] = 8.1554715871500623E+00
    s_corr[71] = 6.8518898508557946E+00
    s_corr[72] = 4.4219094292144900E+00
    s_corr[73] = 1.1115847759264300E+01
    s_corr[74] = 4.8793936123173694E+01
    s_corr[75] = 1.4608640562779471E+00
    s_corr[76] = 7.3335746032859470E+00
    s_corr[77] = 7.3541764115858905E+00
    s_corr[78] = 6.4012488006615680E+00
    s_corr[79] = 8.0927561433097157E+00
    s_corr[80] = 1.5885890375194892E+01
    s_corr[81] = 5.0617824505327755E+00
    
    check1 = results_check.results_check_vec(problem.mp.s, s_corr, problem.mp.comm, tol=tol)
    success = results_check.success_check([check1], problem.mp.comm)
    
    return success



def init():
    
    factor_kPa_mmHg = 7.500615
    
    return {'q_vin_l_0' : 74632.1588103,
            'p_at_l_0' : 0.48051281,
            'q_vout_l_0' : -0.93068153,
            'p_v_l_0' : 0.40588065,
            'p_ar_sys_0' : 9.71269593,
            'q_ar_sys_0' : 58404.0433389,
            'p_arperi_sys_0' : 9.31559213,
            'q_arspl_sys_0' : 18350.1031085,
            'q_arespl_sys_0' : 17211.1248084,
            'q_armsc_sys_0' : 13499.9845749,
            'q_arcer_sys_0' : 9210.19904427,
            'q_arcor_sys_0' : 3070.00584018,
            'p_venspl_sys_0' : 2.38145517,
            'q_venspl_sys_0' : 21808.7362018,
            'p_venespl_sys_0' : 2.40415517,
            'q_venespl_sys_0' : 22345.2886231,
            'p_venmsc_sys_0' : 2.40208803,
            'q_venmsc_sys_0' : 17549.9364684,
            'p_vencer_sys_0' : 2.40683763,
            'q_vencer_sys_0' : 11927.5847843,
            'p_vencor_sys_0' : 2.40697379,
            'q_vencor_sys_0' : 3987.25952259,
            'p_ven_sys_0' : 2.0255366,
            'q_ven_sys_0' : 93493.5104991,
            'p_at_r_0' : 0.23046119,
            'q_vin_r_0' : 69031.9211031,
            'q_vout_r_0' : -0.23518387,
            'p_v_r_0' : 0.16142927,
            'p_ar_pul_0' : 2.51326797,
            'q_ar_pul_0' : 20252.8943157,
            'p_cap_pul_0' : 2.36137126,
            'q_cap_pul_0' : 38454.403139,
            'p_ven_pul_0' : 2.07296324,
            'q_ven_pul_0' : 106163.362138,
            
            'V_alv_0' : 1.0e6,
            'q_alv_0' : 0.0,
            'p_alv_0' : 100.0,
            
            'fCO2_alv_0' : 0.03259099, # 0.05263 # Ben-Tal, J Theor Biol (2006) p. 491
            'fO2_alv_0' : 0.14908848, # 0.1368 # Ben-Tal, J Theor Biol (2006) p. 491
            # initial systemic arterial organ in-fluxes
            'q_arspl_sys_in_0' : 16518.4952469,
            'q_arespl_sys_in_0' : 16521.0063354,
            'q_armsc_sys_in_0' : 12962.1476343,
            'q_arcer_sys_in_0' : 8801.34256122,
            'q_arcor_sys_in_0' : 2953.03385005,
            # initial partial pressures
            'ppCO2_at_r_0' : 15.84975896,
            'ppO2_at_r_0' : 5.06761563,
            'ppCO2_v_r_0' : 15.84965579,
            'ppO2_v_r_0' : 5.0676205,
            'ppCO2_ar_pul_0' : 15.84968334,
            'ppO2_ar_pul_0' : 5.06760886,
            'ppCO2_cap_pul_0' : 3.46305764,
            'ppO2_cap_pul_0' : 13.45820709,
            'ppCO2_ven_pul_0' : 4.1406755,
            'ppO2_ven_pul_0' : 11.95569877,
            'ppCO2_at_l_0' : 4.14587707,
            'ppO2_at_l_0' : 11.94274139,
            'ppCO2_v_l_0' : 4.13625075,
            'ppO2_v_l_0' : 11.95937491,
            'ppCO2_ar_sys_0' : 4.13575614,
            'ppO2_ar_sys_0' : 11.96032172,
            'ppCO2_arspl_sys_0' : 8.15647261,
            'ppO2_arspl_sys_0' : 6.85196469,
            'ppCO2_arespl_sys_0' : 4.42226072,
            'ppO2_arespl_sys_0' : 11.11610101,
            'ppCO2_armsc_sys_0' : 48.79926708,
            'ppO2_armsc_sys_0' : 1.46259561,
            'ppCO2_arcer_sys_0' : 7.33381251,
            'ppO2_arcer_sys_0' : 7.35495005,
            'ppCO2_arcor_sys_0' : 6.40146402,
            'ppO2_arcor_sys_0' : 8.09356649,
            'ppCO2_venspl_sys_0' : 8.15546037,
            'ppO2_venspl_sys_0' : 6.85187768,
            'ppCO2_venespl_sys_0' : 4.42189807,
            'ppO2_venespl_sys_0' : 11.11584935,
            'ppCO2_venmsc_sys_0' : 48.79378298,
            'ppO2_venmsc_sys_0' : 1.46085239,
            'ppCO2_vencer_sys_0' : 7.33356516,
            'ppO2_vencer_sys_0' : 7.3541769,
            'ppCO2_vencor_sys_0' : 6.40123917,
            'ppO2_vencor_sys_0' : 8.092761,
            'ppCO2_ven_sys_0' : 15.8522185,
            'ppO2_ven_sys_0' : 5.06721842}


def param():
    
    # parameters in kg-mm-s-mmol unit system
    
    C_ar_sys = 13081.684615
    R_ar_sys = 7.2e-06
    L_ar_sys = 6.67e-07
    Z_ar_sys = 6e-06

    C_arspl_sys = 378.680344119
    R_arspl_sys = 0.00037788
    C_arespl_sys = 123.931748984
    R_arespl_sys = 0.000401568
    C_armsc_sys = 96.3913603212
    R_armsc_sys = 0.000512112
    C_arcer_sys = 75.7360688238
    R_arcer_sys = 0.00075012
    C_arcor_sys = 20.6552914974
    R_arcor_sys = 0.00225036

    C_venspl_sys = 181766.565177
    R_venspl_sys = 1.632e-05
    C_venespl_sys = 59487.2395125
    R_venespl_sys = 1.6944e-05
    C_venmsc_sys = 46267.8529542
    R_venmsc_sys = 2.1456e-05
    C_vencer_sys = 33048.4663958
    R_vencer_sys = 3.1968e-05
    C_vencor_sys = 9914.53991875
    R_vencor_sys = 9.5664e-05

    C_ven_sys = 82621.1659896
    R_ven_sys = 1.92e-05
    L_ven_sys = 0.0

    C_ar_pul = 10000.0
    R_ar_pul = 7.5e-06
    L_ar_pul = 0.0
    Z_ar_pul = 0.0

    C_cap_pul = 10000.0
    R_cap_pul = 7.5e-06

    C_ven_pul = 50000.0
    R_ven_pul = 1.5e-05
    L_ven_pul = 0.0
    
    t_ed = 0.2
    t_es = 0.53
    
    ### unstressed compartment volumes, diffult to estimate - use literature values!
    # these volumes only become relevant for the gas transport models as they determine the capacity of each
    # compartment to store constituents - however, they are also used for postprocessing of the flow models...
    V_at_l_u = 5000.0 # applies only in case of 0D or prescribed atria
    V_at_r_u = 4000.0 # applies only in case of 0D or prescribed atria
    V_v_l_u = 10000.0 # applies only in case of 0D or prescribed ventricles
    V_v_r_u = 8000.0 # applies only in case of 0D or prescribed ventricles
    V_ar_sys_u = 0.0 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_ar_pul_u = 0.0 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_ven_pul_u = 120.0e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    # peripheral systemic arterial
    V_arspl_sys_u = 274.4e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_arespl_sys_u = 134.64e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_armsc_sys_u = 105.8e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_arcer_sys_u = 72.13e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_arcor_sys_u = 24.0e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    # peripheral systemic venous
    V_venspl_sys_u = 1121.0e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_venespl_sys_u = 550.0e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_venmsc_sys_u = 432.14e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_vencer_sys_u = 294.64e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_vencor_sys_u = 98.21e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3
    V_ven_sys_u = 100.0e3 # estimated (Ursino et al. do not have that extra venous compartment...)
    # pulmonary capillary
    V_cap_pul_u = 123.0e3 # Ursino et al. Am J Physiol Heart Circ Physiol (2000), mm^3


    # external air pressure
    U_m = 1.0e2 # 1 bar = 100 kPa
    #external gas fractions
    fCO2_ext = 0.0004
    fO2_ext = 0.21
    
    # 0D lung
    R_airw = 1.33e-7 # overall resistance of the conducting airways (airways resistance to flow), kPa s/mm^3, Ben-Tal, J Theor Biol (2006) p. 492
    L_alv = 9.87e-10 # alveolar inertance, kPa s^2/mm^3, Rodarte and Rehder (1986)
    R_alv = 0.0 # alveolar resistance, kPa s/mm^3, Rodarte and Rehder (1986)
    E_alv = 3.33e-7 # alveolar elastance, kPa/mm^3, Ben-Tal, J Theor Biol (2006) p. 492
    V_lung_dead = 150.0e3 # dead lung volume, mm^3, Ben-Tal, J Theor Biol (2006) p. 492
    V_lung_u = 0.0 # unstressed lung volume, mm^3, Ben-Tal, J Theor Biol (2006) p. 492
    V_lung_total = 5.0e6#2.5e6 # total alveolar lung volume, mm^3
    
    V_lung_tidal = 600.0e3 # lung tidal volume, mm^3, Ben-Tal, J Theor Biol (2006) p. 492
    
    T_breath = 4.5 # period of one breath, in s
    T_breath = 2.5

    # brething frequency
    omega_breath = 2.*np.pi/T_breath # rad/s, Ben-Tal, J Theor Biol (2006) p. 492

    #tissue volumes
    V_tissspl = 3.243e6 # Christiansen et al. (1996), p. 100 ("liver" + "kidney"), mm^3
    V_tissespl = 0.217e6 # Christiansen et al. (1996), p. 100 ("remaining"), mm^3
    V_tissmsc = 26.773e6 # Christiansen et al. (1996), p. 100, mm^3
    V_tisscer = 1.3e6 # Christiansen et al. (1996), p. 100, mm^3
    V_tisscor = 0.307e6 # Christiansen et al. (1996), p. 100, mm^3
    V_tiss_total = V_tissspl+V_tissespl+V_tissmsc+V_tisscer+V_tisscor

    #CO2 and O2 metabolic rates - in mmol/s
    M_CO2_total_base = 0.193 # mmol/s, cf. Christiansen 1996, p. 91 - 0.192 and 0.103 CANNOT BE (resp. coeff.!!!)
    M_O2_total_base = 0.238 # mmol/s, cf. Christiansen 1996, p. 91 - 0.192 and 0.103 CANNOT BE (resp. coeff.!!!)
    beta_O2 = 1.0e-8 # mmol/mm^3, oxygen concentration when the metabolic rate is half of the maximum value

    M_CO2_total = M_CO2_total_base
    M_O2_total = M_O2_total_base
    
    ### well, some assumption that rates distribute according to tissue volumes...
    M_CO2_arspl = M_CO2_total_base * V_tissspl/V_tiss_total
    M_O2_arspl = M_O2_total_base * V_tissspl/V_tiss_total
    M_CO2_arespl = M_CO2_total_base * V_tissespl/V_tiss_total
    M_O2_arespl = M_O2_total_base * V_tissespl/V_tiss_total
    M_CO2_armsc = M_CO2_total_base * V_tissmsc/V_tiss_total + (M_CO2_total-M_CO2_total_base)
    M_O2_armsc = M_O2_total_base * V_tissmsc/V_tiss_total + (M_O2_total-M_O2_total_base)
    M_CO2_arcer = M_CO2_total_base * V_tisscer/V_tiss_total
    M_O2_arcer = M_O2_total_base * V_tisscer/V_tiss_total
    M_CO2_arcor = M_CO2_total_base * V_tisscor/V_tiss_total
    M_O2_arcor = M_O2_total_base * V_tisscor/V_tiss_total

    # solubility constants for CO2 and O2 in blood (plasma) and tissue
    alpha_CO2 = 24.75e-8 # mmol/(kPa mm^3), Ben-Tal, J Theor Biol (2006) p. 492
    alpha_O2 = 1.05e-8 # mmol/(kPa mm^3), Ben-Tal, J Theor Biol (2006) p. 492; 0.983e-8 acc. to Christiansen (1996), p. 92

    # hemoglobin concentration of the blood, in molar value / volume (default: Christiansen (1996), p. 92, unit: mmol/mm^3)
    c_Hb = 9.3e-6

    # lung diffusion capacities
    kappa_CO2 = 23.7e-2 # lung diffusion capacity of CO2, mmol/(s kPa), Ben-Tal, J Theor Biol (2006) p. 492
    kappa_O2 = 11.7e-2 # lung diffusion capacity of O2, mmol/(s kPa), Ben-Tal, J Theor Biol (2006) p. 492
    
    # oxygen concentration when the metabolic rate is half of the maximum value (Christiansen (1996), p. 52)
    beta_O2 = 1.0e-8
    
    # vapor pressure of water at 37 °C
    # should be 47.1 mmHg = 6.279485 kPa !
    # however we specify it as an input parameter since its decimal power depends on the system of units your whole model is specified in!
    # i.e. if you have kg - mm - s - mmol, it's 6.279485 kPa
    p_vap_water_37 = 6.279485
    
    # molar volume of an ideal gas
    # should be 22.4 liters per mol !
    # however we specify it as an input parameter since its decimal power depends on the system of units your whole model is specified in!
    # i.e. if you have kg - mm - s - mmol, it's 22.4e3 mm^3 / mmol
    V_m_gas = 22.4e3


    return {'R_ar_sys' : R_ar_sys,
            'C_ar_sys' : C_ar_sys,
            'L_ar_sys' : L_ar_sys,
            'Z_ar_sys' : Z_ar_sys,
            'R_arspl_sys' : R_arspl_sys,
            'C_arspl_sys' : C_arspl_sys,
            'R_arespl_sys' : R_arespl_sys,
            'C_arespl_sys' : C_arespl_sys,
            'R_armsc_sys' : R_armsc_sys,
            'C_armsc_sys' : C_armsc_sys,
            'R_arcer_sys' : R_arcer_sys,
            'C_arcer_sys' : C_arcer_sys,
            'R_arcor_sys' : R_arcor_sys,
            'C_arcor_sys' : C_arcor_sys,
            'R_venspl_sys' : R_venspl_sys,
            'C_venspl_sys' : C_venspl_sys,
            'R_venespl_sys' : R_venespl_sys,
            'C_venespl_sys' : C_venespl_sys,
            'R_venmsc_sys' : R_venmsc_sys,
            'C_venmsc_sys' : C_venmsc_sys,
            'R_vencer_sys' : R_vencer_sys,
            'C_vencer_sys' : C_vencer_sys,
            'R_vencor_sys' : R_vencor_sys,
            'C_vencor_sys' : C_vencor_sys,
            'R_ar_pul' : R_ar_pul,
            'C_ar_pul' : C_ar_pul,
            'L_ar_pul' : L_ar_pul,
            'Z_ar_pul' : Z_ar_pul,
            'R_cap_pul' : R_cap_pul,
            'C_cap_pul' : C_cap_pul,
            'R_ven_sys' : R_ven_sys,
            'C_ven_sys' : C_ven_sys, 
            'L_ven_sys' : L_ven_sys,
            'R_ven_pul' : R_ven_pul,
            'C_ven_pul' : C_ven_pul,
            'L_ven_pul' : L_ven_pul,
            # atrial elastances
            'E_at_max_l' : 2.9e-5,
            'E_at_min_l' : 9.0e-6,
            'E_at_max_r' : 1.8e-5,
            'E_at_min_r' : 8.0e-6,
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
            # timings
            't_ed' : t_ed,
            't_es' : t_es,
            'T_cycl' : 1.0,
            # unstressed compartment volumes
            'V_at_l_u' : V_at_l_u,
            'V_at_r_u' : V_at_r_u,
            'V_v_l_u' : V_v_l_u,
            'V_v_r_u' : V_v_r_u,
            'V_ar_sys_u' : V_ar_sys_u,
            'V_arspl_sys_u' : V_arspl_sys_u,
            'V_arespl_sys_u' : V_arespl_sys_u,
            'V_armsc_sys_u' : V_armsc_sys_u,
            'V_arcer_sys_u' : V_arcer_sys_u,
            'V_arcor_sys_u' : V_arcor_sys_u,
            'V_venspl_sys_u' : V_venspl_sys_u,
            'V_venespl_sys_u' : V_venespl_sys_u,
            'V_venmsc_sys_u' : V_venmsc_sys_u,
            'V_vencer_sys_u' : V_vencer_sys_u,
            'V_vencor_sys_u' : V_vencor_sys_u,
            'V_ven_sys_u' : V_ven_sys_u,
            'V_ar_pul_u' : V_ar_pul_u,
            'V_cap_pul_u' : V_cap_pul_u,
            'V_ven_pul_u' : V_ven_pul_u,
            # airway and alveolar parameters
            'R_airw' : R_airw,
            'L_alv' : L_alv,
            'R_alv' : R_alv,
            'E_alv' : E_alv,
            
            'U_m' : U_m,
            'V_lung_dead' : V_lung_dead,
            'V_lung_u' : V_lung_u,
            'V_lung_total' : V_lung_total,
            
            'V_lung_tidal' : V_lung_tidal,
            
            'omega_breath' : omega_breath,
            
            # gas fractions in the atmosphere
            'fCO2_ext' : fCO2_ext,
            'fO2_ext' : fO2_ext,
            
            'V_m_gas' : V_m_gas,
            'p_vap_water_37' : p_vap_water_37,
            
            'kappa_CO2' : kappa_CO2,
            'kappa_O2' : kappa_O2,
            'alpha_CO2' : alpha_CO2,
            'alpha_O2' : alpha_O2,
            # hemoglobin concentration in the blood
            'c_Hb' : c_Hb,
            # consumption and production rates
            'M_CO2_arspl' : M_CO2_arspl,
            'M_O2_arspl' : M_O2_arspl,
            'M_CO2_arespl' : M_CO2_arespl,
            'M_O2_arespl' : M_O2_arespl,
            'M_CO2_armsc' : M_CO2_armsc,
            'M_O2_armsc' : M_O2_armsc,
            'M_CO2_arcer' : M_CO2_arcer,
            'M_O2_arcer' : M_O2_arcer,
            'M_CO2_arcor' : M_CO2_arcor,
            'M_O2_arcor' : M_O2_arcor,

            'beta_O2' :  beta_O2,
            
            # tissue volumes
            'V_tissspl' : V_tissspl,
            'V_tissespl' : V_tissespl,
            'V_tissmsc' : V_tissmsc,
            'V_tisscer' : V_tisscer,
            'V_tisscor' : V_tisscor}



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
