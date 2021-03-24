#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import sys, os, subprocess, time
import math
import numpy as np
import sympy as sp

from cardiovascular0D import cardiovascular0Dbase
from mpiroutines import allgather_vec

# systemic and pulmonary closed-loop circulation model including capillary flow, each heart chamber can be treated individually,
# either as 0D elastance model, volume or flux coming from a 3D solid, or interface fluxes from a 3D fluid model

# 34 governing equations (uncomment and paste directly into a LaTeX environment):

## left heart and systemic circulation:
#\begin{align}
#&-Q_{\mathrm{at}}^{\ell} = q_{\mathrm{ven}}^{\mathrm{pul}} - q_{\mathrm{v,in}}^{\ell}\nonumber\\
#&\tilde{R}_{\mathrm{v,in}}^{\ell} q_{\mathrm{v,in}}^{\ell} = p_{\mathrm{at}}^{\ell}-p_{\mathrm{v}}^{\ell}\nonumber\\
#&-Q_{\mathrm{v}}^{\ell} = q_{\mathrm{v,in}}^{\ell} - q_{\mathrm{v,out}}^{\ell}\nonumber\\
#&\tilde{R}_{\mathrm{v,out}}^{\ell} q_{\mathrm{v,out}}^{\ell} = p_{\mathrm{v}}^{\ell}-p_{\mathrm{ar}}^{\mathrm{sys}}\nonumber\\
#&C_{\mathrm{ar}}^{\mathrm{sys}} \left(\frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t} - Z_{\mathrm{ar}}^{\mathrm{sys}}\frac{\mathrm{d}q_{\mathrm{v,out}}^{\ell}}{\mathrm{d}t}\right) = q_{\mathrm{v,out}}^{\ell} - q_{\mathrm{ar}}^{\mathrm{sys}}\nonumber\\
#&L_{\mathrm{ar}}^{\mathrm{sys}}\frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{sys}}q_{\mathrm{ar}}^{\mathrm{sys}}=p_{\mathrm{ar}}^{\mathrm{sys}} - Z_{\mathrm{ar}}^{\mathrm{sys}}q_{\mathrm{v,out}}^{\ell}-p_{\mathrm{ar,peri}}^{\mathrm{sys}}\nonumber\\
#&\left(\sum_{j\in\{\mathrm{spl,espl,\atop msc,cer,cor}\}}\!\!\!\!\!\!\!\!\!C_{\mathrm{ar},j}^{\mathrm{sys}}\right) \frac{\mathrm{d}p_{\mathrm{ar,peri}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{sys}}-\!\!\!\!\!\sum_{j\in\{\mathrm{spl,espl,\atop msc,cer,cor}\}}\!\!\!\!\!\!\!\!\!q_{\mathrm{ar},j}^{\mathrm{sys}}\nonumber\\
#&R_{\mathrm{ar},i}^{\mathrm{sys}} q_{\mathrm{ar},i}^{\mathrm{sys}} = p_{\mathrm{ar,peri}}^{\mathrm{sys}} - p_{\mathrm{ven},i}^{\mathrm{sys}}, \quad\scriptstyle{i\in\{\mathrm{spl,espl,\atop msc,cer,cor}\}}\nonumber\\
#&C_{\mathrm{ven},i}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven},i}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar},i}^{\mathrm{sys}} - q_{\mathrm{ven},i}^{\mathrm{sys}}, \quad\scriptstyle{i\in\{\mathrm{spl,espl,\atop msc,cer,cor}\}}\nonumber\\
#&R_{\mathrm{ven},i}^{\mathrm{sys}} q_{\mathrm{ven},i}^{\mathrm{sys}} = p_{\mathrm{ven},i}^{\mathrm{sys}}-p_{\mathrm{ven}}^{\mathrm{sys}}, \quad\scriptstyle{i\in\{\mathrm{spl,espl,\atop msc,cer,cor}\}}\nonumber\\
#&C_{\mathrm{ven}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{sys}}}{\mathrm{d}t} = \!\!\!\!\sum_{j=\mathrm{spl,espl,\atop msc,cer,cor}}\!\!\!\!\!q_{\mathrm{ven},j}^{\mathrm{sys}}-q_{\mathrm{ven}}^{\mathrm{sys}}\nonumber\\
#&L_{\mathrm{ven}}^{\mathrm{sys}}\frac{\mathrm{d}q_{\mathrm{ven}}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ven}}^{\mathrm{sys}} q_{\mathrm{ven}}^{\mathrm{sys}} = p_{\mathrm{ven}}^{\mathrm{sys}} - p_{\mathrm{at}}^{r}\nonumber
#\end{align}

## right heart and pulmonary circulation:
#\begin{align}
#&-Q_{\mathrm{at}}^{r} = q_{\mathrm{ven}}^{\mathrm{sys}} - q_{\mathrm{v,in}}^{r}\nonumber\\
#&\tilde{R}_{\mathrm{v,in}}^{r}q_{\mathrm{v,in}}^{r} = p_{\mathrm{at}}^{r}-p_{\mathrm{v}}^{r}\nonumber\\
#&-Q_{\mathrm{v}}^{r} = q_{\mathrm{v,in}}^{r} - q_{\mathrm{v,out}}^{r}\nonumber\\
#&\tilde{R}_{\mathrm{v,out}}^{r} q_{\mathrm{v,out}}^{r} = p_{\mathrm{v}}^{r}-p_{\mathrm{ar}}^{\mathrm{pul}}\nonumber\\
#&C_{\mathrm{ar}}^{\mathrm{pul}} \left(\frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} - Z_{\mathrm{ar}}^{\mathrm{pul}}\frac{\mathrm{d}q_{\mathrm{v,out}}^{r}}{\mathrm{d}t}\right) = q_{\mathrm{v,out}}^{r} - q_{\mathrm{ar}}^{\mathrm{pul}}\nonumber\\
#&L_{\mathrm{ar}}^{\mathrm{pul}}\frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{pul}} q_{\mathrm{ar}}^{\mathrm{pul}}=p_{\mathrm{ar}}^{\mathrm{pul}} - Z_{\mathrm{ar}}^{\mathrm{pul}}q_{\mathrm{v,out}}^{r}-p_{\mathrm{cap}}^{\mathrm{pul}}\nonumber\\
#&C_{\mathrm{cap}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{cap}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{pul}} - q_{\mathrm{cap}}^{\mathrm{pul}}\nonumber\\
#&R_{\mathrm{cap}}^{\mathrm{pul}} q_{\mathrm{cap}}^{\mathrm{pul}}=p_{\mathrm{cap}}^{\mathrm{pul}}-p_{\mathrm{ven}}^{\mathrm{pul}}\nonumber\\
#&C_{\mathrm{ven}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{cap}}^{\mathrm{pul}} - q_{\mathrm{ven}}^{\mathrm{pul}}\nonumber\\
#&L_{\mathrm{ven}}^{\mathrm{pul}}\frac{\mathrm{d}q_{\mathrm{ven}}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ven}}^{\mathrm{pul}} q_{\mathrm{ven}}^{\mathrm{pul}}=p_{\mathrm{ven}}^{\mathrm{pul}}-p_{\mathrm{at}}^{\ell}\nonumber
#\end{align}

class cardiovascular0Dsyspulcap(cardiovascular0Dbase):
    
    def __init__(self, theta, params, chmodels, prescrpath=None, have_elast=False, cq='volume', valvelaw=['pwlin_pres',0], comm=None):
        # initialize base class
        cardiovascular0Dbase.__init__(self, theta, comm=comm)
        
        # parameters
        # circulatory system parameters: resistances (R), compliances (C), inertances (L), impedances (Z)
        self.R_ar_sys = params['R_ar_sys']
        self.C_ar_sys = params['C_ar_sys']
        self.L_ar_sys = params['L_ar_sys']
        self.Z_ar_sys = params['Z_ar_sys']

        # peripheral arterial compliances and resistances
        self.R_arspl_sys = params['R_arspl_sys']
        self.C_arspl_sys = params['C_arspl_sys']
        self.R_arespl_sys = params['R_arespl_sys']
        self.C_arespl_sys = params['C_arespl_sys']
        self.R_armsc_sys = params['R_armsc_sys']
        self.C_armsc_sys = params['C_armsc_sys']
        self.R_arcer_sys = params['R_arcer_sys']
        self.C_arcer_sys = params['C_arcer_sys']
        self.R_arcor_sys = params['R_arcor_sys']
        self.C_arcor_sys = params['C_arcor_sys']
        # peripheral venous compliances and resistances
        self.R_venspl_sys = params['R_venspl_sys']
        self.C_venspl_sys = params['C_venspl_sys']
        self.R_venespl_sys = params['R_venespl_sys']
        self.C_venespl_sys = params['C_venespl_sys']
        self.R_venmsc_sys = params['R_venmsc_sys']
        self.C_venmsc_sys = params['C_venmsc_sys']
        self.R_vencer_sys = params['R_vencer_sys']
        self.C_vencer_sys = params['C_vencer_sys']
        self.R_vencor_sys = params['R_vencor_sys']
        self.C_vencor_sys = params['C_vencor_sys']
        
        self.R_ar_pul = params['R_ar_pul']
        self.C_ar_pul = params['C_ar_pul']
        self.L_ar_pul = params['L_ar_pul']
        self.Z_ar_pul = params['Z_ar_pul']
        # pulmonary capillary compliance and resistance
        self.R_cap_pul = params['R_cap_pul']
        self.C_cap_pul = params['C_cap_pul']
        
        self.R_ven_sys = params['R_ven_sys']
        self.C_ven_sys = params['C_ven_sys']
        self.L_ven_sys = params['L_ven_sys']
        self.R_ven_pul = params['R_ven_pul']
        self.C_ven_pul = params['C_ven_pul']
        self.L_ven_pul = params['L_ven_pul']
        
        # ventricular elastances (for 0D ventricles)
        self.E_v_max_l = params['E_v_max_l']
        self.E_v_min_l = params['E_v_min_l']
        self.E_v_max_r = params['E_v_max_r']
        self.E_v_min_r = params['E_v_min_r']

        # atrial elastances (for 0D atria)
        self.E_at_max_l = params['E_at_max_l']
        self.E_at_min_l = params['E_at_min_l']
        self.E_at_max_r = params['E_at_max_r']
        self.E_at_min_r = params['E_at_min_r']

        # valve resistances
        self.R_vin_l_min = params['R_vin_l_min']
        self.R_vin_l_max = params['R_vin_l_max']
        self.R_vin_r_min = params['R_vin_r_min']
        self.R_vin_r_max = params['R_vin_r_max']
        self.R_vout_l_min = params['R_vout_l_min']
        self.R_vout_l_max = params['R_vout_l_max']
        self.R_vout_r_min = params['R_vout_r_min']
        self.R_vout_r_max = params['R_vout_r_max']
        
        # end-diastolic and end-systolic timings
        self.t_ed = params['t_ed']
        self.t_es = params['t_es']
        self.T_cycl = params['T_cycl']
        
        # unstressed compartment volumes (for post-processing)
        self.V_at_l_u = params['V_at_l_u']
        self.V_at_r_u = params['V_at_r_u']
        self.V_v_l_u = params['V_v_l_u']
        self.V_v_r_u = params['V_v_r_u']
        self.V_ar_sys_u = params['V_ar_sys_u']
        self.V_arspl_sys_u = params['V_arspl_sys_u']
        self.V_arespl_sys_u = params['V_arespl_sys_u']
        self.V_armsc_sys_u = params['V_armsc_sys_u']
        self.V_arcer_sys_u = params['V_arcer_sys_u']
        self.V_arcor_sys_u = params['V_arcor_sys_u']
        self.V_venspl_sys_u = params['V_venspl_sys_u']
        self.V_venespl_sys_u = params['V_venespl_sys_u']
        self.V_venmsc_sys_u = params['V_venmsc_sys_u']
        self.V_vencer_sys_u = params['V_vencer_sys_u']
        self.V_vencor_sys_u = params['V_vencor_sys_u']
        self.V_ven_sys_u = params['V_ven_sys_u']
        self.V_ar_pul_u = params['V_ar_pul_u']
        self.V_cap_pul_u = params['V_cap_pul_u']
        self.V_ven_pul_u = params['V_ven_pul_u']
        
        self.chmodels = chmodels
        self.valvelaw = valvelaw[0]
        self.epsvalve = valvelaw[1]
        
        self.prescrpath = prescrpath
        self.have_elast = have_elast
        
        self.cq = cq

        # set up arrays
        self.setup_arrays()
        
        # setup chambers
        self.set_chamber_interfaces()

        # set up symbolic equations
        self.equation_map()
        
        # symbolic stiffness matrix
        self.set_stiffness()

        # make Lambda functions out of symbolic expressions
        self.lambdify_expressions()



    def setup_arrays(self):

        # number of degrees of freedom
        self.numdof = 34

        self.elastarrays = [[]]*4
        
        self.si, self.switch_V, self.switch_p = [0]*4, [1]*4, [0]*4 # default values

        self.vindex_ch = [3,27,1,25] # coupling variable indices (decreased by 1 for pressure coupling!)
        self.vname_prfx, self.cname_prfx = ['p']*4, ['V']*4 # default values
    
        # set those ids which are relevant for monolithic direct coupling
        self.v_ids, self.c_ids = [], []
        self.cindex_ch = [2,26,0,24]
    
        self.set_solve_arrays()


    def evaluate(self, x, dt, t, df=None, f=None, K=None, c=[], y=[], a=None):
        
        fnc = self.evaluate_chamber_state(y, t)

        cardiovascular0Dbase.evaluate(self, x, dt, t, df, f, K, c, y, a, fnc)

    
    def equation_map(self):
        
        self.varmap={'q_vin_l' : 0+self.si[2], ''+self.vname_prfx[2]+'_at_l' : 1-self.si[2], 'q_vout_l' : 2+self.si[0], ''+self.vname_prfx[0]+'_v_l' : 3-self.si[0], 'p_ar_sys' : 4, 'q_ar_sys' : 5, 'p_arperi_sys' : 6, 'q_arspl_sys' : 7, 'q_arespl_sys' : 8, 'q_armsc_sys' : 9, 'q_arcer_sys' : 10, 'q_arcor_sys' : 11, 'p_venspl_sys' : 12, 'q_venspl_sys' : 13, 'p_venespl_sys' : 14, 'q_venespl_sys' : 15, 'p_venmsc_sys' : 16, 'q_venmsc_sys' : 17, 'p_vencer_sys' : 18, 'q_vencer_sys' : 19, 'p_vencor_sys' : 20, 'q_vencor_sys' : 21, 'p_ven_sys' : 22, 'q_ven_sys' : 23, 'q_vin_r' : 24+self.si[3], ''+self.vname_prfx[3]+'_at_r' : 25-self.si[3], 'q_vout_r' : 26+self.si[1], ''+self.vname_prfx[1]+'_v_r' : 27-self.si[1], 'p_ar_pul' : 28, 'q_ar_pul' : 29, 'p_cap_pul' : 30,'q_cap_pul' : 31, 'p_ven_pul' : 32, 'q_ven_pul' : 33}
        self.auxmap={''+self.cname_prfx[2]+'_at_l' : 0, ''+self.cname_prfx[0]+'_v_l' : 2, 'V_ar_sys' : 4, 'V_arperi_sys' : 6, 'V_venspl_sys' : 12, 'V_venespl_sys' : 14, 'V_venmsc_sys' : 16, 'V_vencer_sys' : 18, 'V_vencor_sys' : 20, 'V_ven_sys' : 22, ''+self.cname_prfx[3]+'_at_r' : 24, ''+self.cname_prfx[1]+'_v_r' : 26, 'V_ar_pul' : 28, 'V_cap_pul' : 30, 'V_ven_pul' : 32}

        if self.chmodels['lv']['type']=='3D_fem' and self.chmodels['lv']['interfaces'] > 1: self.auxmap['p_v_l_d'] = 3
        if self.chmodels['rv']['type']=='3D_fem' and self.chmodels['rv']['interfaces'] > 1: self.auxmap['p_v_r_d'] = 27
        if self.chmodels['la']['type']=='3D_fem' and self.chmodels['la']['interfaces'] > 1: self.auxmap['p_at_l_d'] = 1
        if self.chmodels['ra']['type']=='3D_fem' and self.chmodels['ra']['interfaces'] > 1: self.auxmap['p_at_r_d'] = 25


        self.t_            = sp.Symbol('t_')
        q_vin_l_           = sp.Symbol('q_vin_l_')
        p_at_l_, p_at_l_d_ = sp.Symbol('p_at_l_'), sp.Symbol('p_at_l_d_')
        q_vout_l_          = sp.Symbol('q_vout_l_')
        p_v_l_, p_v_l_d_   = sp.Symbol('p_v_l_'), sp.Symbol('p_v_l_d_')
        p_ar_sys_          = sp.Symbol('p_ar_sys_')
        q_ar_sys_          = sp.Symbol('q_ar_sys_')
        p_arperi_sys_      = sp.Symbol('p_arperi_sys_')
        q_arspl_sys_       = sp.Symbol('q_arspl_sys_')
        q_arespl_sys_      = sp.Symbol('q_arespl_sys_')
        q_armsc_sys_       = sp.Symbol('q_armsc_sys_')
        q_arcer_sys_       = sp.Symbol('q_arcer_sys_')
        q_arcor_sys_       = sp.Symbol('q_arcor_sys_')
        p_venspl_sys_      = sp.Symbol('p_venspl_sys_')
        q_venspl_sys_      = sp.Symbol('q_venspl_sys_')
        p_venespl_sys_     = sp.Symbol('p_venespl_sys_')
        q_venespl_sys_     = sp.Symbol('q_venespl_sys_')
        p_venmsc_sys_      = sp.Symbol('p_venmsc_sys_')
        q_venmsc_sys_      = sp.Symbol('q_venmsc_sys_')
        p_vencer_sys_      = sp.Symbol('p_vencer_sys_')
        q_vencer_sys_      = sp.Symbol('q_vencer_sys_')
        p_vencor_sys_      = sp.Symbol('p_vencor_sys_')
        q_vencor_sys_      = sp.Symbol('q_vencor_sys_')
        p_ven_sys_         = sp.Symbol('p_ven_sys_')
        q_ven_sys_         = sp.Symbol('q_ven_sys_')
        q_vin_r_           = sp.Symbol('q_vin_r_')
        p_at_r_, p_at_r_d_ = sp.Symbol('p_at_r_'), sp.Symbol('p_at_r_d_')
        q_vout_r_          = sp.Symbol('q_vout_r_')
        p_v_r_, p_v_r_d_   = sp.Symbol('p_v_r_'), sp.Symbol('p_v_r_d_')
        p_ar_pul_          = sp.Symbol('p_ar_pul_')
        q_ar_pul_          = sp.Symbol('q_ar_pul_')
        p_cap_pul_         = sp.Symbol('p_cap_pul_')
        q_cap_pul_         = sp.Symbol('q_cap_pul_')
        p_ven_pul_         = sp.Symbol('p_ven_pul_')
        q_ven_pul_         = sp.Symbol('q_ven_pul_')
        VQ_v_l_            = sp.Symbol('VQ_v_l_')
        VQ_v_r_            = sp.Symbol('VQ_v_r_')
        VQ_at_l_           = sp.Symbol('VQ_at_l_')
        VQ_at_r_           = sp.Symbol('VQ_at_r_')
        
        E_v_l_             = sp.Symbol('E_v_l_')
        E_v_r_             = sp.Symbol('E_v_r_')
        E_at_l_            = sp.Symbol('E_at_l_')
        E_at_r_            = sp.Symbol('E_at_r_')
        
        # dofs to differentiate w.r.t.
        self.x_[0+self.si[2]] = q_vin_l_
        self.x_[1-self.si[2]] = p_at_l_
        self.x_[2+self.si[0]] = q_vout_l_
        self.x_[3-self.si[0]] = p_v_l_
        self.x_[4] = p_ar_sys_
        self.x_[5] = q_ar_sys_
        self.x_[6] = p_arperi_sys_
        self.x_[7] = q_arspl_sys_
        self.x_[8] = q_arespl_sys_
        self.x_[9] = q_armsc_sys_
        self.x_[10] = q_arcer_sys_
        self.x_[11] = q_arcor_sys_
        self.x_[12] = p_venspl_sys_
        self.x_[13] = q_venspl_sys_
        self.x_[14] = p_venespl_sys_
        self.x_[15] = q_venespl_sys_
        self.x_[16] = p_venmsc_sys_
        self.x_[17] = q_venmsc_sys_
        self.x_[18] = p_vencer_sys_
        self.x_[19] = q_vencer_sys_
        self.x_[20] = p_vencor_sys_
        self.x_[21] = q_vencor_sys_
        self.x_[22] = p_ven_sys_
        self.x_[23] = q_ven_sys_
        self.x_[24+self.si[3]] = q_vin_r_
        self.x_[25-self.si[3]] = p_at_r_
        self.x_[26+self.si[1]] = q_vout_r_
        self.x_[27-self.si[1]] = p_v_r_
        self.x_[28] = p_ar_pul_
        self.x_[29] = q_ar_pul_
        self.x_[30] = p_cap_pul_
        self.x_[31] = q_cap_pul_
        self.x_[32] = p_ven_pul_
        self.x_[33] = q_ven_pul_
        
        # set chamber states and variables (e.g., express V in terms of p and E in case of elastance models, ...)
        VQ_v_l_, p_v_l_, p_v_l_d_ = self.set_coupling_state('lv', {'vq' : VQ_v_l_, 'p' : p_v_l_, 'pdown' : p_v_l_d_}, [E_v_l_])
        VQ_v_r_, p_v_r_, p_v_r_d_ = self.set_coupling_state('rv', {'vq' : VQ_v_r_, 'p' : p_v_r_, 'pdown' : p_v_r_d_}, [E_v_r_])
        VQ_at_l_, p_at_l_, p_at_l_d_ = self.set_coupling_state('la', {'vq' : VQ_at_l_, 'p' : p_at_l_, 'pdown' : p_at_l_d_}, [E_at_l_])
        VQ_at_r_, p_at_r_, p_at_r_d_ = self.set_coupling_state('ra', {'vq' : VQ_at_r_, 'p' : p_at_r_, 'pdown' : p_at_r_d_}, [E_at_r_])

        # set valve resistances
        R_vin_l_, R_vin_r_, R_vout_l_, R_vout_r_ = self.set_valve_resistances(p_v_l_,p_v_l_d_,p_v_r_,p_v_r_d_,p_at_l_d_,p_at_r_d_,p_ar_sys_,p_ar_pul_)


        # df part of rhs contribution (df - df_old)/dt
        self.df_[0]  = VQ_at_l_ * self.switch_V[2]
        self.df_[1]  = 0
        self.df_[2]  = VQ_v_l_ * self.switch_V[0]
        self.df_[3]  = 0
        self.df_[4]  = self.C_ar_sys * (p_ar_sys_ - self.Z_ar_sys * q_vout_l_)
        self.df_[5]  = (self.L_ar_sys/self.R_ar_sys) * q_ar_sys_
        self.df_[6]  = (self.C_arspl_sys+self.C_arespl_sys+self.C_armsc_sys+self.C_arcer_sys+self.C_arcor_sys) * p_arperi_sys_
        self.df_[7]  = 0
        self.df_[8]  = 0
        self.df_[9]  = 0
        self.df_[10] = 0
        self.df_[11] = 0
        self.df_[12] = self.C_venspl_sys * p_venspl_sys_
        self.df_[13] = 0
        self.df_[14] = self.C_venespl_sys * p_venespl_sys_
        self.df_[15] = 0
        self.df_[16] = self.C_venmsc_sys * p_venmsc_sys_
        self.df_[17] = 0
        self.df_[18] = self.C_vencer_sys * p_vencer_sys_
        self.df_[19] = 0
        self.df_[20] = self.C_vencor_sys * p_vencor_sys_
        self.df_[21] = 0
        self.df_[22] = self.C_ven_sys * p_ven_sys_
        self.df_[23] = (self.L_ven_sys/self.R_ven_sys) * q_ven_sys_
                # -----------------------------------------------------------
        self.df_[24] = VQ_at_r_ * self.switch_V[3]
        self.df_[25] = 0
        self.df_[26] = VQ_v_r_ * self.switch_V[1]
        self.df_[27] = 0
        self.df_[28] = self.C_ar_pul * (p_ar_pul_ - self.Z_ar_pul * q_vout_r_)
        self.df_[29] = (self.L_ar_pul/self.R_ar_pul) * q_ar_pul_
        self.df_[30] = self.C_cap_pul * p_cap_pul_
        self.df_[31] = 0
        self.df_[32] = self.C_ven_pul * p_ven_pul_
        self.df_[33] = (self.L_ven_pul/self.R_ven_pul) * q_ven_pul_

        # f part of rhs contribution theta * f + (1-theta) * f_old
        self.f_[0]  = -q_ven_pul_ + q_vin_l_ - (1-self.switch_V[2]) * VQ_at_l_
        self.f_[1]  = (p_at_l_d_-p_v_l_)/R_vin_l_ - q_vin_l_
        self.f_[2]  = -q_vin_l_ + q_vout_l_ - (1-self.switch_V[0]) * VQ_v_l_
        self.f_[3]  = (p_v_l_d_-p_ar_sys_)/R_vout_l_ - q_vout_l_
        self.f_[4]  = -q_vout_l_ + q_ar_sys_
        self.f_[5]  = (p_arperi_sys_ - p_ar_sys_ + self.Z_ar_sys * q_vout_l_)/self.R_ar_sys + q_ar_sys_
        self.f_[6]  = -q_ar_sys_ + (q_arspl_sys_ + q_arespl_sys_ + q_armsc_sys_ + q_arcer_sys_ + q_arcor_sys_)
        self.f_[7]  = (p_venspl_sys_ - p_arperi_sys_)/self.R_arspl_sys + q_arspl_sys_
        self.f_[8]  = (p_venespl_sys_ - p_arperi_sys_)/self.R_arespl_sys + q_arespl_sys_
        self.f_[9]  = (p_venmsc_sys_ - p_arperi_sys_)/self.R_armsc_sys + q_armsc_sys_
        self.f_[10] = (p_vencer_sys_ - p_arperi_sys_)/self.R_arcer_sys + q_arcer_sys_
        self.f_[11] = (p_vencor_sys_ - p_arperi_sys_)/self.R_arcor_sys + q_arcor_sys_
        self.f_[12] = q_venspl_sys_ - q_arspl_sys_
        self.f_[13] = (p_ven_sys_ - p_venspl_sys_)/self.R_venspl_sys + q_venspl_sys_
        self.f_[14] = q_venespl_sys_ - q_arespl_sys_
        self.f_[15] = (p_ven_sys_ - p_venespl_sys_)/self.R_venespl_sys + q_venespl_sys_
        self.f_[16] = q_venmsc_sys_ - q_armsc_sys_
        self.f_[17] = (p_ven_sys_ - p_venmsc_sys_)/self.R_venmsc_sys + q_venmsc_sys_
        self.f_[18] = q_vencer_sys_ - q_arcer_sys_
        self.f_[19] = (p_ven_sys_ - p_vencer_sys_)/self.R_vencer_sys + q_vencer_sys_
        self.f_[20] = q_vencor_sys_ - q_arcor_sys_
        self.f_[21] = (p_ven_sys_ - p_vencor_sys_)/self.R_vencor_sys + q_vencor_sys_
        self.f_[22] = q_ven_sys_ - (q_venspl_sys_ + q_venespl_sys_ + q_venmsc_sys_ + q_vencer_sys_ + q_vencor_sys_)
        self.f_[23] = (p_at_r_ - p_ven_sys_)/self.R_ven_sys + q_ven_sys_
                # -----------------------------------------------------------
        self.f_[24] = -q_ven_sys_ + q_vin_r_ - (1-self.switch_V[3]) * VQ_at_r_
        self.f_[25] = (p_at_r_d_-p_v_r_)/R_vin_r_ - q_vin_r_
        self.f_[26] = -q_vin_r_ + q_vout_r_ - (1-self.switch_V[1]) * VQ_v_r_
        self.f_[27] = (p_v_r_d_-p_ar_pul_)/R_vout_r_ - q_vout_r_
        self.f_[28] = -q_vout_r_ + q_ar_pul_
        self.f_[29] = (p_cap_pul_ - p_ar_pul_ + self.Z_ar_pul * q_vout_r_)/self.R_ar_pul + q_ar_pul_
        self.f_[30] = -q_ar_pul_ + q_cap_pul_
        self.f_[31] = (p_ven_pul_ - p_cap_pul_)/self.R_cap_pul + q_cap_pul_
        self.f_[32] = -q_cap_pul_ + q_ven_pul_
        self.f_[33] = (p_at_l_ - p_ven_pul_)/self.R_ven_pul + q_ven_pul_    


        # auxiliary variable vector (mainly in order to store quantities for post-processing)
        self.a_[0]  = VQ_at_l_ * (1-self.switch_p[2]) + p_at_l_ * self.switch_p[2]
        self.a_[1]  = p_at_l_d_ * self.switch_p[2]
        self.a_[2]  = VQ_v_l_ * (1-self.switch_p[0]) + p_v_l_ * self.switch_p[0]
        self.a_[3]  = p_v_l_d_ * self.switch_p[0]
        self.a_[4]  = self.C_ar_sys * (p_ar_sys_ - self.Z_ar_sys * q_vout_l_) + self.V_ar_sys_u
        self.a_[5]  = 0.
        self.a_[6]  = (self.C_arspl_sys+self.C_arespl_sys+self.C_armsc_sys+self.C_arcer_sys+self.C_arcor_sys) * p_arperi_sys_ + self.V_arspl_sys_u+self.V_arespl_sys_u+self.V_armsc_sys_u+self.V_arcer_sys_u+self.V_arcor_sys_u
        self.a_[7]  = 0.
        self.a_[8]  = 0.
        self.a_[9]  = 0.
        self.a_[10] = 0.
        self.a_[11] = 0.
        self.a_[12] = self.C_venspl_sys * p_venspl_sys_ + self.V_venspl_sys_u
        self.a_[13] = 0.
        self.a_[14] = self.C_venespl_sys * p_venespl_sys_ + self.V_venespl_sys_u
        self.a_[15] = 0.
        self.a_[16] = self.C_venmsc_sys * p_venmsc_sys_ + self.V_venmsc_sys_u
        self.a_[17] = 0.
        self.a_[18] = self.C_vencer_sys * p_vencer_sys_ + self.V_vencer_sys_u
        self.a_[19] = 0.
        self.a_[20] = self.C_vencor_sys * p_vencor_sys_ + self.V_vencor_sys_u
        self.a_[21] = 0.
        self.a_[22] = self.C_ven_sys * p_ven_sys_ + self.V_ven_sys_u
        self.a_[23] = 0.
                # -----------------------------------------------------------
        self.a_[24] = VQ_at_r_ * (1-self.switch_p[3]) + p_at_r_ * self.switch_p[3]
        self.a_[25] = p_at_r_d_ * self.switch_p[3]
        self.a_[26] = VQ_v_r_ * (1-self.switch_p[1]) + p_v_r_ * self.switch_p[1]
        self.a_[27] = p_v_r_d_ * self.switch_p[1]
        self.a_[28] = self.C_ar_pul * (p_ar_pul_ - self.Z_ar_pul * q_vout_r_) + self.V_ar_pul_u
        self.a_[29] = 0.
        self.a_[30] = self.C_cap_pul * p_cap_pul_ + self.V_cap_pul_u
        self.a_[31] = 0.
        self.a_[32] = self.C_ven_pul * p_ven_pul_ + self.V_ven_pul_u
        self.a_[33] = 0.




    def initialize(self, var, iniparam):

        var[0+self.si[2]] = iniparam['q_vin_l_0']
        var[1-self.si[2]] = iniparam[''+self.vname_prfx[2]+'_at_l_0']
        var[2+self.si[0]] = iniparam['q_vout_l_0']
        var[3-self.si[0]] = iniparam[''+self.vname_prfx[0]+'_v_l_0']
        var[4]  = iniparam['p_ar_sys_0']
        var[5]  = iniparam['q_ar_sys_0']

        var[6]  = iniparam['p_arperi_sys_0']
        var[7]  = iniparam['q_arspl_sys_0']
        var[8]  = iniparam['q_arespl_sys_0']
        var[9]  = iniparam['q_armsc_sys_0']
        var[10] = iniparam['q_arcer_sys_0']
        var[11] = iniparam['q_arcor_sys_0']
        var[12] = iniparam['p_venspl_sys_0']
        var[13] = iniparam['q_venspl_sys_0']
        var[14] = iniparam['p_venespl_sys_0']
        var[15] = iniparam['q_venespl_sys_0']
        var[16] = iniparam['p_venmsc_sys_0']
        var[17] = iniparam['q_venmsc_sys_0']
        var[18] = iniparam['p_vencer_sys_0']
        var[19] = iniparam['q_vencer_sys_0']
        var[20] = iniparam['p_vencor_sys_0']
        var[21] = iniparam['q_vencor_sys_0']

        var[22] = iniparam['p_ven_sys_0']
        var[23] = iniparam['q_ven_sys_0']
        var[24+self.si[3]] = iniparam['q_vin_r_0']
        var[25-self.si[3]] = iniparam[''+self.vname_prfx[3]+'_at_r_0']
        var[26+self.si[1]] = iniparam['q_vout_r_0']
        var[27-self.si[1]] = iniparam[''+self.vname_prfx[1]+'_v_r_0']
        var[28] = iniparam['p_ar_pul_0']
        var[29] = iniparam['q_ar_pul_0']
        var[30] = iniparam['p_cap_pul_0']
        var[31] = iniparam['q_cap_pul_0']
        var[32] = iniparam['p_ven_pul_0']
        var[33] = iniparam['q_ven_pul_0']



    def check_periodic(self, varTc, varTc_old, eps, check, cyclerr):
        
        if isinstance(varTc, np.ndarray): varTc_sq, varTc_old_sq = varTc, varTc_old
        else: varTc_sq, varTc_old_sq = allgather_vec(varTc, self.comm), allgather_vec(varTc_old, self.comm)

        if check=='allvar':
            
            vals = []
            for i in range(len(varTc_sq)):
                vals.append( math.fabs((varTc_sq[i]-varTc_old_sq[i])/max(1.0,math.fabs(varTc_old_sq[i]))) )

        elif check=='pQvar':
            
            vals = []
            pQvar_ids = [1-self.si[2],3-self.si[0],4,6,12,14,16,18,20,22,25-self.si[3],27-self.si[1],28,30,32]
            for i in range(len(varTc_sq)):
                if i in pQvar_ids:
                    vals.append( math.fabs((varTc_sq[i]-varTc_old_sq[i])/max(1.0,math.fabs(varTc_old_sq[i]))) )

        else:
            
            raise NameError("Unknown check option!")

        cyclerr[0] = max(vals)

        if cyclerr[0] <= eps:
            is_periodic = True
        else:
            is_periodic = False
            
        return is_periodic



    def print_to_screen(self, var, aux):
        
        if isinstance(var, np.ndarray): var_sq = var
        else: var_sq = allgather_vec(var, self.comm)

        if self.comm.rank == 0:
            
            print("Output of 0D vascular model (syspulcap):")
            
            print('{:<12s}{:<3s}{:<10.1f}{:<3s}{:<9s}{:<3s}{:<10.1f}'.format(''+self.cname_prfx[2]+'_at_l',' = ',aux[0],'   ',''+self.cname_prfx[3]+'_at_r',' = ',aux[24]))
            print('{:<12s}{:<3s}{:<10.1f}{:<3s}{:<9s}{:<3s}{:<10.1f}'.format(''+self.cname_prfx[0]+'_v_l',' = ',aux[2],'   ',''+self.cname_prfx[1]+'_v_r',' = ',aux[26]))
            
            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format(''+self.vname_prfx[2]+'_at_l',' = ',var_sq[1-self.si[2]],'   ',''+self.vname_prfx[3]+'_at_r',' = ',var_sq[25-self.si[3]]))
            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format(''+self.vname_prfx[0]+'_v_l',' = ',var_sq[3-self.si[0]],'   ',''+self.vname_prfx[1]+'_v_r',' = ',var_sq[27-self.si[1]]))
            
            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format('p_ar_sys',' = ',var_sq[4],'   ','p_ar_pul',' = ',var_sq[28]))
            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format('p_arperi_sys',' = ',var_sq[6],'   ','p_cap_pul',' = ',var_sq[30]))
            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format('p_ven_sys',' = ',var_sq[22],'   ','p_ven_pul',' = ',var_sq[32]))

            sys.stdout.flush()





def postprocess_groups_syspulcap(groups, indpertaftercyl=0, multiscalegandr=False):
    
    # index 0
    groups.append({'pres_time_sys_l'  : ['p_at_l', 'p_v_l', 'p_ar_sys', 'p_arperi_sys', 'p_venspl_sys', 'p_venespl_sys', 'p_venmsc_sys', 'p_vencer_sys', 'p_vencor_sys', 'p_ven_sys'],
                'tex'                 : ['$p_{\\\mathrm{at}}^{\\\ell}$', '$p_{\\\mathrm{v}}^{\\\ell}$', '$p_{\\\mathrm{ar}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ar,peri}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ven,spl}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ven,espl}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ven,msc}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ven,cer}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ven,cor}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ven}}^{\\\mathrm{sys}}$'],
                'lines'               : [1, 2, 3, 4, 10, 11, 12, 13, 14, 15]})
    # index 1
    groups.append({'pres_time_pul_r'  : ['p_at_r', 'p_v_r', 'p_ar_pul', 'p_cap_pul', 'p_ven_pul'],
                'tex'                 : ['$p_{\\\mathrm{at}}^{r}$', '$p_{\\\mathrm{v}}^{r}$', '$p_{\\\mathrm{ar}}^{\\\mathrm{pul}}$', '$p_{\\\mathrm{cap}}^{\\\mathrm{pul}}$', '$p_{\\\mathrm{ven}}^{\\\mathrm{pul}}$'],
                'lines'               : [16, 17, 18, 19, 20]})
    # index 2
    groups.append({'flux_time_sys_l'  : ['q_vin_l', 'q_vout_l', 'q_ar_sys', 'q_arspl_sys', 'q_arespl_sys', 'q_armsc_sys', 'q_arcer_sys', 'q_arcor_sys', 'q_venspl_sys', 'q_venespl_sys', 'q_venmsc_sys', 'q_vencer_sys', 'q_vencor_sys', 'q_ven_sys'],
                'tex'                 : ['$q_{\\\mathrm{v,in}}^{\\\ell}$', '$q_{\\\mathrm{v,out}}^{\\\ell}$', '$q_{\\\mathrm{ar}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ar,spl}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ar,espl}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ar,msc}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ar,cer}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ar,cor}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ven,spl}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ven,espl}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ven,msc}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ven,cer}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ven,cor}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ven}}^{\\\mathrm{sys}}$'],
                'lines'               : [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]})
    # index 3
    groups.append({'flux_time_pul_r'  : ['q_vin_r', 'q_vout_r', 'q_ar_pul', 'q_cap_pul', 'q_ven_pul'],
                'tex'                 : ['$q_{\\\mathrm{v,in}}^{r}$', '$q_{\\\mathrm{v,out}}^{r}$', '$q_{\\\mathrm{ar}}^{\\\mathrm{pul}}$', '$q_{\\\mathrm{cap}}^{\\\mathrm{pul}}$', '$q_{\\\mathrm{ven}}^{\\\mathrm{pul}}$'],
                'lines'               : [16, 17, 18, 19, 20]})
    # index 4
    groups.append({'vol_time_l_r'     : ['V_at_l', 'V_v_l', 'V_at_r', 'V_v_r'],
                'tex'                 : ['$V_{\\\mathrm{at}}^{\\\ell}$', '$V_{\\\mathrm{v}}^{\\\ell}$', '$V_{\\\mathrm{at}}^{r}$', '$V_{\\\mathrm{v}}^{r}$'],
                'lines'               : [1, 2, 16, 17]})
    # index 5
    groups.append({'vol_time_compart' : ['V_at_l', 'V_v_l', 'V_at_r', 'V_v_r', 'V_ar_sys', 'V_arperi_sys', 'V_venspl_sys', 'V_venespl_sys', 'V_venmsc_sys', 'V_vencer_sys', 'V_vencor_sys', 'V_ven_sys', 'V_ar_pul', 'V_cap_pul', 'V_ven_pul', 'V_all'],
                'tex'                 : ['$V_{\\\mathrm{at}}^{\\\ell}$', '$V_{\\\mathrm{v}}^{\\\ell}$', '$V_{\\\mathrm{at}}^{r}$', '$V_{\\\mathrm{v}}^{r}$', '$V_{\\\mathrm{ar}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ar,peri}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven,spl}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven,espl}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven,msc}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven,cer}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven,cor}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ar}}^{\\\mathrm{pul}}$', '$V_{\\\mathrm{cap}}^{\\\mathrm{pul}}$', '$V_{\\\mathrm{ven}}^{\\\mathrm{pul}}$', '$\\\sum V$'],
                'lines'               : [1, 2, 16, 17, 3, 4, 10, 11, 12, 13, 14, 15, 18, 19, 20, 99]})
    
    # pv loops are only considered for the last cycle
    
    if indpertaftercyl > 0: # for comparison of healthy/baseline and perturbed states
        if multiscalegandr:
            # index 6
            groups.append({'pres_vol_v_l_r_PERIODIC'  : ['pV_v_l_gandr', 'pV_v_r_gandr', 'pV_v_l_last', 'pV_v_r_last', 'pV_v_l_baseline', 'pV_v_r_baseline'],
                        'tex'                      : ['$p_{\\\mathrm{v}}^{\\\ell,\\\mathrm{G\\&R}}$', '$p_{\\\mathrm{v}}^{r,\\\mathrm{G\\&R}}$', '$p_{\\\mathrm{v}}^{\\\ell}$', '$p_{\\\mathrm{v}}^{r}$', '$p_{\\\mathrm{v}}^{\\\ell,\\\mathrm{ref}}$', '$p_{\\\mathrm{v}}^{r,\\\mathrm{ref}}$'],
                        'lines'                    : [21, 22, 102, 117, 97, 98]})
            # index 7
            groups.append({'pres_vol_at_l_r_PERIODIC' : ['pV_at_l_gandr', 'pV_at_r_gandr', 'pV_at_l_last', 'pV_at_r_last', 'pV_at_l_baseline', 'pV_at_r_baseline'],
                        'tex'                      : ['$p_{\\\mathrm{at}}^{\\\ell,\\\mathrm{G\\&R}}$', '$p_{\\\mathrm{at}}^{r,\\\mathrm{G\\&R}}$', '$p_{\\\mathrm{at}}^{\\\ell}$', '$p_{\\\mathrm{at}}^{r}$', '$p_{\\\mathrm{at}}^{\\\ell,\\\mathrm{ref}}$', '$p_{\\\mathrm{at}}^{r,\\\mathrm{ref}}$'],
                        'lines'                    : [23, 24, 101, 116, 97, 98]})
        else:
            # index 6
            groups.append({'pres_vol_v_l_r_PERIODIC'  : ['pV_v_l_last', 'pV_v_r_last', 'pV_v_l_baseline', 'pV_v_r_baseline'],
                        'tex'                      : ['$p_{\\\mathrm{v}}^{\\\ell}$', '$p_{\\\mathrm{v}}^{r}$', '$p_{\\\mathrm{v}}^{\\\ell,\\\mathrm{ref}}$', '$p_{\\\mathrm{v}}^{r,\\\mathrm{ref}}$'],
                        'lines'                    : [2, 17, 97, 98]})
            # index 7
            groups.append({'pres_vol_at_l_r_PERIODIC' : ['pV_at_l_last', 'pV_at_r_last', 'pV_at_l_baseline', 'pV_at_r_baseline'],
                        'tex'                      : ['$p_{\\\mathrm{at}}^{\\\ell}$', '$p_{\\\mathrm{at}}^{r}$', '$p_{\\\mathrm{at}}^{\\\ell,\\\mathrm{ref}}$', '$p_{\\\mathrm{at}}^{r,\\\mathrm{ref}}$'],
                        'lines'                    : [1, 16, 97, 98]})
    else:
        # index 6
        groups.append({'pres_vol_v_l_r_PERIODIC'  : ['pV_v_l_last', 'pV_v_r_last'],
                       'tex'                      : ['$p_{\\\mathrm{v}}^{\\\ell}$', '$p_{\\\mathrm{v}}^{r}$'],
                       'lines'                    : [2, 17]})
        # index 7
        groups.append({'pres_vol_at_l_r_PERIODIC' : ['pV_at_l_last', 'pV_at_r_last'],
                       'tex'                      : ['$p_{\\\mathrm{at}}^{\\\ell}$', '$p_{\\\mathrm{at}}^{r}$'],
                       'lines'                    : [1, 16]})
    
    
    # now append all the values again but with suffix PERIODIC, since we want to plot both:
    # values over all heart cycles as well as only for the periodic cycle

    # index 8
    groups.append({'pres_time_sys_l_PERIODIC'  : list(groups[0].values())[0],
                   'tex'                       : list(groups[0].values())[1],
                   'lines'                     : list(groups[0].values())[2]})
    # index 9
    groups.append({'pres_time_pul_r_PERIODIC'  : list(groups[1].values())[0],
                   'tex'                       : list(groups[1].values())[1],
                   'lines'                     : list(groups[1].values())[2]})
    # index 10
    groups.append({'flux_time_sys_l_PERIODIC'  : list(groups[2].values())[0],
                   'tex'                       : list(groups[2].values())[1],
                   'lines'                     : list(groups[2].values())[2]})
    # index 11    
    groups.append({'flux_time_pul_r_PERIODIC'  : list(groups[3].values())[0],
                   'tex'                       : list(groups[3].values())[1],
                   'lines'                     : list(groups[3].values())[2]})
    # index 12        
    groups.append({'vol_time_l_r_PERIODIC'     : list(groups[4].values())[0],
                   'tex'                       : list(groups[4].values())[1],
                   'lines'                     : list(groups[4].values())[2]})
    # index 13            
    groups.append({'vol_time_compart_PERIODIC' : list(groups[5].values())[0],
                   'tex'                       : list(groups[5].values())[1],
                   'lines'                     : list(groups[5].values())[2]})







# similar to syspulcap model, however with the coronaries branching off after the aortic valve
# and directly feeding back into the right atrium

class cardiovascular0Dsyspulcapcor_veins(cardiovascular0Dsyspulcap):
    
    def setup_arrays(self):
        
        # number of degrees of freedom
        self.numdof = 38

        self.elastarrays = [[]]*4
        
        self.si, self.switch_V, self.switch_p = [0]*4, [1]*4, [0]*4 # default values

        self.vindex_ch = [3,28,1,26] # coupling variable indices (decreased by 1 for pressure coupling!)
        self.vname_prfx, self.cname_prfx = ['p']*4, ['V']*4 # default values
    
        # set those ids which are relevant for monolithic direct coupling
        self.v_ids, self.c_ids = [], []
        self.cindex_ch = [2,27,0,25]
    
        self.set_solve_arrays()

    
    def equation_map(self):
        
        self.varmap={'q_vin_l' : 0+self.si[2], ''+self.vname_prfx[2]+'_at_l' : 1-self.si[2], 'q_vout_l' : 2+self.si[0], ''+self.vname_prfx[0]+'_v_l' : 3-self.si[0], 'p_ar_sys' : 4, 'q_arcor_sys' : 5, 'q_ar_sys' : 6, 'p_arperi_sys' : 7, 'q_arspl_sys' : 8, 'q_arespl_sys' : 9, 'q_armsc_sys' : 10, 'q_arcer_sys' : 11, 'p_venspl_sys' : 12, 'q_venspl_sys' : 13, 'p_venespl_sys' : 14, 'q_venespl_sys' : 15, 'p_venmsc_sys' : 16, 'q_venmsc_sys' : 17, 'p_vencer_sys' : 18, 'q_vencer_sys' : 19, 'p_vencor_sys' : 20, 'q_vencor_sys' : 21, 'p_ven_sys' : 22, 'q_ven1_sys' : 23, 'q_ven2_sys' : 24, 'q_vin_r' : 25+self.si[3], ''+self.vname_prfx[3]+'_at_r' : 26-self.si[3], 'q_vout_r' : 27+self.si[1], ''+self.vname_prfx[1]+'_v_r' : 28-self.si[1], 'p_ar_pul' : 29, 'q_ar_pul' : 30, 'p_cap_pul' : 31,'q_cap_pul' : 32, 'p_ven_pul' : 33, 'q_ven1_pul' : 34, 'q_ven2_pul' : 35, 'q_ven3_pul' : 36, 'q_ven4_pul' : 37}
        self.auxmap={''+self.cname_prfx[2]+'_at_l' : 0, ''+self.cname_prfx[0]+'_v_l' : 2, 'V_ar_arcor_sys' : 4, 'V_arperi_sys' : 7, 'V_venspl_sys' : 12, 'V_venespl_sys' : 14, 'V_venmsc_sys' : 16, 'V_vencer_sys' : 18, 'V_vencor_sys' : 20, 'V_ven_sys' : 22, ''+self.cname_prfx[3]+'_at_r' : 25, ''+self.cname_prfx[1]+'_v_r' : 27, 'V_ar_pul' : 29, 'V_cap_pul' : 31, 'V_ven_pul' : 33}

        if self.chmodels['lv']['type']=='3D_fem' and self.chmodels['lv']['interfaces'] > 1: self.auxmap['p_v_l_d'] = 3
        if self.chmodels['rv']['type']=='3D_fem' and self.chmodels['rv']['interfaces'] > 1: self.auxmap['p_v_r_d'] = 28
        if self.chmodels['la']['type']=='3D_fem' and self.chmodels['la']['interfaces'] > 1: self.auxmap['p_at_l_d'] = 1
        if self.chmodels['ra']['type']=='3D_fem' and self.chmodels['ra']['interfaces'] > 1: self.auxmap['p_at_r_d'] = 26
        

        self.t_            = sp.Symbol('t_')
        q_vin_l_           = sp.Symbol('q_vin_l_')
        p_at1_l_, p_at2_l_, p_at3_l_, p_at4_l_, p_at_l_d_ = sp.Symbol('p_at1_l_'), sp.Symbol('p_at2_l_'), sp.Symbol('p_at3_l_'), sp.Symbol('p_at4_l_'), sp.Symbol('p_at_l_d_')
        q_vout_l_          = sp.Symbol('q_vout_l_')
        p_v_l_, p_v_l_d_   = sp.Symbol('p_v_l_'), sp.Symbol('p_v_l_d_')
        p_ar_sys_          = sp.Symbol('p_ar_sys_')
        q_arcor_sys_       = sp.Symbol('q_arcor_sys_')
        q_ar_sys_          = sp.Symbol('q_ar_sys_')
        p_arperi_sys_      = sp.Symbol('p_arperi_sys_')
        q_arspl_sys_       = sp.Symbol('q_arspl_sys_')
        q_arespl_sys_      = sp.Symbol('q_arespl_sys_')
        q_armsc_sys_       = sp.Symbol('q_armsc_sys_')
        q_arcer_sys_       = sp.Symbol('q_arcer_sys_')
        p_venspl_sys_      = sp.Symbol('p_venspl_sys_')
        q_venspl_sys_      = sp.Symbol('q_venspl_sys_')
        p_venespl_sys_     = sp.Symbol('p_venespl_sys_')
        q_venespl_sys_     = sp.Symbol('q_venespl_sys_')
        p_venmsc_sys_      = sp.Symbol('p_venmsc_sys_')
        q_venmsc_sys_      = sp.Symbol('q_venmsc_sys_')
        p_vencer_sys_      = sp.Symbol('p_vencer_sys_')
        q_vencer_sys_      = sp.Symbol('q_vencer_sys_')
        p_vencor_sys_      = sp.Symbol('p_vencor_sys_')
        q_vencor_sys_      = sp.Symbol('q_vencor_sys_')
        p_ven_sys_         = sp.Symbol('p_ven_sys_')
        q_ven1_sys_        = sp.Symbol('q_ven1_sys_')
        q_ven2_sys_        = sp.Symbol('q_ven2_sys_')
        q_vin_r_           = sp.Symbol('q_vin_r_')
        p_at1_r_, p_at2_r_, p_at_r_d_ = sp.Symbol('p_at1_r_'), sp.Symbol('p_at2_r_'), sp.Symbol('p_at_r_d_')
        q_vout_r_          = sp.Symbol('q_vout_r_')
        p_v_r_, p_v_r_d_   = sp.Symbol('p_v_r_'), sp.Symbol('p_v_r_d_')
        p_ar_pul_          = sp.Symbol('p_ar_pul_')
        q_ar_pul_          = sp.Symbol('q_ar_pul_')
        p_cap_pul_         = sp.Symbol('p_cap_pul_')
        q_cap_pul_         = sp.Symbol('q_cap_pul_')
        p_ven_pul_         = sp.Symbol('p_ven_pul_')
        q_ven1_pul_        = sp.Symbol('q_ven1_pul_')
        q_ven2_pul_        = sp.Symbol('q_ven2_pul_')
        q_ven3_pul_        = sp.Symbol('q_ven3_pul_')
        q_ven4_pul_        = sp.Symbol('q_ven4_pul_')
        VQ_v_l_            = sp.Symbol('VQ_v_l_')
        VQ_v_r_            = sp.Symbol('VQ_v_r_')
        VQ_at_l_           = sp.Symbol('VQ_at_l_')
        VQ_at_r_           = sp.Symbol('VQ_at_r_')
        
        E_v_l_             = sp.Symbol('E_v_l_')
        E_v_r_             = sp.Symbol('E_v_r_')
        E_at_l_            = sp.Symbol('E_at_l_')
        E_at_r_            = sp.Symbol('E_at_r_')
        
        # dofs to differentiate w.r.t.
        self.x_[0+self.si[2]] = q_vin_l_
        self.x_[1-self.si[2]] = p_at1_l_
        self.x_[2+self.si[0]] = q_vout_l_
        self.x_[3-self.si[0]] = p_v_l_
        self.x_[4] = p_ar_sys_
        self.x_[5] = q_arcor_sys_
        self.x_[6] = q_ar_sys_
        self.x_[7] = p_arperi_sys_
        self.x_[8] = q_arspl_sys_
        self.x_[9] = q_arespl_sys_
        self.x_[10] = q_armsc_sys_
        self.x_[11] = q_arcer_sys_
        self.x_[12] = p_venspl_sys_
        self.x_[13] = q_venspl_sys_
        self.x_[14] = p_venespl_sys_
        self.x_[15] = q_venespl_sys_
        self.x_[16] = p_venmsc_sys_
        self.x_[17] = q_venmsc_sys_
        self.x_[18] = p_vencer_sys_
        self.x_[19] = q_vencer_sys_
        self.x_[20] = p_vencor_sys_
        self.x_[21] = q_vencor_sys_
        self.x_[22] = p_ven_sys_
        self.x_[23] = q_ven1_sys_
        self.x_[24] = q_ven2_sys_
        self.x_[25+self.si[3]] = q_vin_r_
        self.x_[26-self.si[3]] = p_at1_r_
        self.x_[27+self.si[1]] = q_vout_r_
        self.x_[28-self.si[1]] = p_v_r_
        self.x_[29] = p_ar_pul_
        self.x_[30] = q_ar_pul_
        self.x_[31] = p_cap_pul_
        self.x_[32] = q_cap_pul_
        self.x_[33] = p_ven_pul_
        self.x_[34] = q_ven1_pul_
        self.x_[35] = q_ven2_pul_
        self.x_[36] = q_ven3_pul_
        self.x_[37] = q_ven4_pul_
        
        # set chamber states and variables (e.g., express V in terms of p and E in case of elastance models, ...)
        VQ_v_l_, p_v_l_, p_v_l_d_ = self.set_coupling_state('lv', {'vq' : VQ_v_l_, 'p' : p_v_l_, 'pdown' : p_v_l_d_}, [E_v_l_])
        VQ_v_r_, p_v_r_, p_v_r_d_ = self.set_coupling_state('rv', {'vq' : VQ_v_r_, 'p' : p_v_r_, 'pdown' : p_v_r_d_}, [E_v_r_])
        VQ_at_l_, p_at1_l_, p_at2_l_, p_at3_l_, p_at4_l_, p_at_l_d_ = self.set_coupling_state('la', {'vq' : VQ_at_l_, 'p' : p_at1_l_, 'p2' : p_at2_l_, 'p3' : p_at3_l_, 'p4' : p_at4_l_, 'pdown' : p_at_l_d_}, [E_at_l_])
        VQ_at_r_, p_at1_r_, p_at2_r_, p_at_r_d_ = self.set_coupling_state('ra', {'vq' : VQ_at_r_, 'p' : p_at1_r_, 'p2' : p_at2_r_, 'pdown' : p_at_r_d_}, [E_at_r_])

        # set valve resistances
        R_vin_l_, R_vin_r_, R_vout_l_, R_vout_r_ = self.set_valve_resistances(p_v_l_,p_v_l_d_,p_v_r_,p_v_r_d_,p_at_l_d_,p_at_r_d_,p_ar_sys_,p_ar_pul_)

        # parallel venous resistances and inertances:
        # assume that the total venous resistance/inertance distributes equally over all two systemic / four pulmonary veins that enter the right / left atrium
        # resistance/inertance in parallel: 1/R_total = 1/R_1 + 1/R_2 + ... + 1/R_n, 1/L_total = 1/L_1 + 1/L_2 + ... + 1/L_n
        # let's say: R_1 = R_2 = ... = R_n, L_1 = L_2 = ... = L_n
        R_ven1_sys, R_ven2_sys = 2.*self.R_ven_sys, 2.*self.R_ven_sys
        R_ven1_pul, R_ven2_pul, R_ven3_pul, R_ven4_pul = 4.*self.R_ven_pul, 4.*self.R_ven_pul, 4.*self.R_ven_pul, 4.*self.R_ven_pul
        L_ven1_sys, L_ven2_sys = 2.*self.L_ven_sys, 2.*self.L_ven_sys
        L_ven1_pul, L_ven2_pul, L_ven3_pul, L_ven4_pul = 4.*self.L_ven_pul, 4.*self.L_ven_pul, 4.*self.L_ven_pul, 4.*self.L_ven_pul


        # df part of rhs contribution (df - df_old)/dt
        self.df_[0]  = VQ_at_l_ * self.switch_V[2]
        self.df_[1]  = 0.
        self.df_[2]  = VQ_v_l_ * self.switch_V[0]
        self.df_[3]  = 0.
        self.df_[4]  = (self.C_ar_sys + self.C_arcor_sys) * p_ar_sys_ - self.C_ar_sys*self.Z_ar_sys * (q_vout_l_ - q_arcor_sys_)
        self.df_[5]  = 0.
        self.df_[6]  = (self.L_ar_sys/self.R_ar_sys) * q_ar_sys_
        self.df_[7]  = (self.C_arspl_sys+self.C_arespl_sys+self.C_armsc_sys+self.C_arcer_sys) * p_arperi_sys_
        self.df_[8]  = 0.
        self.df_[9]  = 0.
        self.df_[10] = 0.
        self.df_[11] = 0.
        self.df_[12] = self.C_venspl_sys * p_venspl_sys_
        self.df_[13] = 0.
        self.df_[14] = self.C_venespl_sys * p_venespl_sys_
        self.df_[15] = 0.
        self.df_[16] = self.C_venmsc_sys * p_venmsc_sys_
        self.df_[17] = 0.
        self.df_[18] = self.C_vencer_sys * p_vencer_sys_
        self.df_[19] = 0.
        self.df_[20] = self.C_vencor_sys * p_vencor_sys_
        self.df_[21] = 0.
        self.df_[22] = self.C_ven_sys * p_ven_sys_
        self.df_[23] = (L_ven1_sys/R_ven1_sys) * q_ven1_sys_
        self.df_[24] = (L_ven2_sys/R_ven2_sys) * q_ven2_sys_
                # -----------------------------------------------------------
        self.df_[25] = VQ_at_r_ * self.switch_V[3]
        self.df_[26] = 0.
        self.df_[27] = VQ_v_r_ * self.switch_V[1]
        self.df_[28] = 0.
        self.df_[29] = self.C_ar_pul * (p_ar_pul_ - self.Z_ar_pul * q_vout_r_)
        self.df_[30] = (self.L_ar_pul/self.R_ar_pul) * q_ar_pul_
        self.df_[31] = self.C_cap_pul * p_cap_pul_
        self.df_[32] = 0.
        self.df_[33] = self.C_ven_pul * p_ven_pul_
        self.df_[34] = (L_ven1_pul/R_ven1_pul) * q_ven1_pul_
        self.df_[35] = (L_ven2_pul/R_ven2_pul) * q_ven2_pul_
        self.df_[36] = (L_ven3_pul/R_ven3_pul) * q_ven3_pul_
        self.df_[37] = (L_ven4_pul/R_ven4_pul) * q_ven4_pul_


        # f part of rhs contribution theta * f + (1-theta) * f_old
        self.f_[0]  = -(q_ven1_pul_ + q_ven2_pul_ + q_ven3_pul_ + q_ven4_pul_) + q_vin_l_ - (1-self.switch_V[2]) * VQ_at_l_
        self.f_[1]  = (p_at_l_d_-p_v_l_)/R_vin_l_ - q_vin_l_
        self.f_[2]  = -q_vin_l_ + q_vout_l_ - (1-self.switch_V[0]) * VQ_v_l_
        self.f_[3]  = (p_v_l_d_-p_ar_sys_)/R_vout_l_ - q_vout_l_
        self.f_[4]  = -q_vout_l_ + q_ar_sys_ + q_arcor_sys_
        self.f_[5]  = (p_vencor_sys_ - p_ar_sys_)/self.R_arcor_sys + q_arcor_sys_
        self.f_[6]  = (p_arperi_sys_ - p_ar_sys_ + self.Z_ar_sys * (q_vout_l_-q_arcor_sys_))/self.R_ar_sys + q_ar_sys_
        self.f_[7]  = -q_ar_sys_ + (q_arspl_sys_ + q_arespl_sys_ + q_armsc_sys_ + q_arcer_sys_)
        self.f_[8]  = (p_venspl_sys_ - p_arperi_sys_)/self.R_arspl_sys + q_arspl_sys_
        self.f_[9]  = (p_venespl_sys_ - p_arperi_sys_)/self.R_arespl_sys + q_arespl_sys_
        self.f_[10] = (p_venmsc_sys_ - p_arperi_sys_)/self.R_armsc_sys + q_armsc_sys_
        self.f_[11] = (p_vencer_sys_ - p_arperi_sys_)/self.R_arcer_sys + q_arcer_sys_
        self.f_[12] = q_venspl_sys_ - q_arspl_sys_
        self.f_[13] = (p_ven_sys_ - p_venspl_sys_)/self.R_venspl_sys + q_venspl_sys_
        self.f_[14] = q_venespl_sys_ - q_arespl_sys_
        self.f_[15] = (p_ven_sys_ - p_venespl_sys_)/self.R_venespl_sys + q_venespl_sys_
        self.f_[16] = q_venmsc_sys_ - q_armsc_sys_
        self.f_[17] = (p_ven_sys_ - p_venmsc_sys_)/self.R_venmsc_sys + q_venmsc_sys_
        self.f_[18] = q_vencer_sys_ - q_arcer_sys_
        self.f_[19] = (p_ven_sys_ - p_vencer_sys_)/self.R_vencer_sys + q_vencer_sys_
        self.f_[20] = q_vencor_sys_ - q_arcor_sys_
        self.f_[21] = (p_at1_r_ - p_vencor_sys_)/self.R_vencor_sys + q_vencor_sys_
        self.f_[22] = (q_ven1_sys_ + q_ven2_sys_) - (q_venspl_sys_ + q_venespl_sys_ + q_venmsc_sys_ + q_vencer_sys_)
        self.f_[23] = (p_at1_r_ - p_ven_sys_)/R_ven1_sys + q_ven1_sys_
        self.f_[24] = (p_at2_r_ - p_ven_sys_)/R_ven2_sys + q_ven2_sys_
                # -----------------------------------------------------------
        self.f_[25] = -(q_ven1_sys_ + q_ven2_sys_) - q_vencor_sys_ + q_vin_r_ - (1-self.switch_V[3]) * VQ_at_r_
        self.f_[26] = (p_at_r_d_-p_v_r_)/R_vin_r_ - q_vin_r_
        self.f_[27] = -q_vin_r_ + q_vout_r_ - (1-self.switch_V[1]) * VQ_v_r_
        self.f_[28] = (p_v_r_d_-p_ar_pul_)/R_vout_r_ - q_vout_r_
        self.f_[29] = -q_vout_r_ + q_ar_pul_
        self.f_[30] = (p_cap_pul_ - p_ar_pul_ + self.Z_ar_pul * q_vout_r_)/self.R_ar_pul + q_ar_pul_
        self.f_[31] = -q_ar_pul_ + q_cap_pul_
        self.f_[32] = (p_ven_pul_ - p_cap_pul_)/self.R_cap_pul + q_cap_pul_
        self.f_[33] = -q_cap_pul_ + (q_ven1_pul_ + q_ven2_pul_ + q_ven3_pul_ + q_ven4_pul_)
        self.f_[34] = (p_at1_l_ - p_ven_pul_)/R_ven1_pul + q_ven1_pul_
        self.f_[35] = (p_at2_l_ - p_ven_pul_)/R_ven2_pul + q_ven2_pul_
        self.f_[36] = (p_at3_l_ - p_ven_pul_)/R_ven3_pul + q_ven3_pul_
        self.f_[37] = (p_at4_l_ - p_ven_pul_)/R_ven4_pul + q_ven4_pul_
    
    
        # auxiliary variable vector (mainly in order to store quantities for post-processing)
        self.a_[0]  = VQ_at_l_ * (1-self.switch_p[2]) + p_at1_l_ * self.switch_p[2]
        self.a_[1]  = p_at_l_d_ * self.switch_p[2]
        self.a_[2]  = VQ_v_l_ * (1-self.switch_p[0]) + p_v_l_ * self.switch_p[0]
        self.a_[3]  = p_v_l_d_ * self.switch_p[0]
        self.a_[4]  = (self.C_ar_sys + self.C_arcor_sys) * p_ar_sys_ - self.C_ar_sys*self.Z_ar_sys * (q_vout_l_ - q_arcor_sys_) + self.V_ar_sys_u + self.V_arcor_sys_u
        self.a_[5]  = 0.
        self.a_[6]  = 0.
        self.a_[7]  = (self.C_arspl_sys+self.C_arespl_sys+self.C_armsc_sys+self.C_arcer_sys) * p_arperi_sys_ + self.V_arspl_sys_u+self.V_arespl_sys_u+self.V_armsc_sys_u+self.V_arcer_sys_u
        self.a_[8]  = 0.
        self.a_[9]  = 0.
        self.a_[10] = 0.
        self.a_[11] = 0.
        self.a_[12] = self.C_venspl_sys * p_venspl_sys_ + self.V_venspl_sys_u
        self.a_[13] = 0.
        self.a_[14] = self.C_venespl_sys * p_venespl_sys_ + self.V_venespl_sys_u
        self.a_[15] = 0.
        self.a_[16] = self.C_venmsc_sys * p_venmsc_sys_ + self.V_venmsc_sys_u
        self.a_[17] = 0.
        self.a_[18] = self.C_vencer_sys * p_vencer_sys_ + self.V_vencer_sys_u
        self.a_[19] = 0.
        self.a_[20] = self.C_vencor_sys * p_vencor_sys_ + self.V_vencor_sys_u
        self.a_[21] = 0.
        self.a_[22] = self.C_ven_sys * p_ven_sys_ + self.V_ven_sys_u
        self.a_[23] = 0.
        self.a_[24] = 0.
                # -----------------------------------------------------------
        self.a_[25] = VQ_at_r_ * (1-self.switch_p[3]) + p_at1_r_ * self.switch_p[3]
        self.a_[26] = p_at_r_d_ * self.switch_p[3]
        self.a_[27] = VQ_v_r_ * (1-self.switch_p[1]) + p_v_r_ * self.switch_p[1]
        self.a_[28] = p_v_r_d_ * self.switch_p[1]
        self.a_[29] = self.C_ar_pul * (p_ar_pul_ - self.Z_ar_pul * q_vout_r_) + self.V_ar_pul_u
        self.a_[30] = 0.
        self.a_[31] = self.C_cap_pul * p_cap_pul_ + self.V_cap_pul_u
        self.a_[32] = 0.
        self.a_[33] = self.C_ven_pul * p_ven_pul_ + self.V_ven_pul_u
        self.a_[34] = 0.
        self.a_[35] = 0.
        self.a_[36] = 0.
        self.a_[37] = 0.



    def initialize(self, var, iniparam):

        var[0+self.si[2]] = iniparam['q_vin_l_0']
        var[1-self.si[2]] = iniparam[''+self.vname_prfx[2]+'_at_l_0']
        var[2+self.si[0]] = iniparam['q_vout_l_0']
        var[3-self.si[0]] = iniparam[''+self.vname_prfx[0]+'_v_l_0']
        var[4]  = iniparam['p_ar_sys_0']
        var[5]  = iniparam['q_arcor_sys_0']
        var[6]  = iniparam['q_ar_sys_0']
        
        var[7]  = iniparam['p_arperi_sys_0']
        var[8]  = iniparam['q_arspl_sys_0']
        var[9]  = iniparam['q_arespl_sys_0']
        var[10] = iniparam['q_armsc_sys_0']
        var[11] = iniparam['q_arcer_sys_0']
        var[12] = iniparam['p_venspl_sys_0']
        var[13] = iniparam['q_venspl_sys_0']
        var[14] = iniparam['p_venespl_sys_0']
        var[15] = iniparam['q_venespl_sys_0']
        var[16] = iniparam['p_venmsc_sys_0']
        var[17] = iniparam['q_venmsc_sys_0']
        var[18] = iniparam['p_vencer_sys_0']
        var[19] = iniparam['q_vencer_sys_0']
        var[20] = iniparam['p_vencor_sys_0']
        var[21] = iniparam['q_vencor_sys_0']

        var[22] = iniparam['p_ven_sys_0']
        var[23] = iniparam['q_ven1_sys_0']
        var[24] = iniparam['q_ven2_sys_0']
        var[25+self.si[3]] = iniparam['q_vin_r_0']
        var[26-self.si[3]] = iniparam[''+self.vname_prfx[3]+'_at_r_0']
        var[27+self.si[1]] = iniparam['q_vout_r_0']
        var[28-self.si[1]] = iniparam[''+self.vname_prfx[1]+'_v_r_0']
        var[29] = iniparam['p_ar_pul_0']
        var[30] = iniparam['q_ar_pul_0']
        var[31] = iniparam['p_cap_pul_0']
        var[32] = iniparam['q_cap_pul_0']
        var[33] = iniparam['p_ven_pul_0']
        var[34] = iniparam['q_ven1_pul_0']
        var[35] = iniparam['q_ven2_pul_0']
        var[36] = iniparam['q_ven3_pul_0']
        var[37] = iniparam['q_ven4_pul_0']



    def check_periodic(self, varTc, varTc_old, eps, check, cyclerr):
        
        if isinstance(varTc, np.ndarray): varTc_sq, varTc_old_sq = varTc, varTc_old
        else: varTc_sq, varTc_old_sq = allgather_vec(varTc, self.comm), allgather_vec(varTc_old, self.comm)

        if check=='allvar':
            
            vals = []
            for i in range(len(varTc_sq)):
                vals.append( math.fabs((varTc_sq[i]-varTc_old_sq[i])/max(1.0,math.fabs(varTc_old_sq[i]))) )

        elif check=='pQvar':
            
            vals = []
            pQvar_ids = [1-self.si[2],3-self.si[0],4,7,12,14,16,18,20,22,26-self.si[3],28-self.si[1],29,31,33]
            for i in range(len(varTc_sq)):
                if i in pQvar_ids:
                    vals.append( math.fabs((varTc_sq[i]-varTc_old_sq[i])/max(1.0,math.fabs(varTc_old_sq[i]))) )

        else:
            
            raise NameError("Unknown check option!")

        cyclerr[0] = max(vals)

        if cyclerr[0] <= eps:
            is_periodic = True
        else:
            is_periodic = False
            
        return is_periodic



    def print_to_screen(self, var, aux):
        
        if isinstance(var, np.ndarray): var_sq = var
        else: var_sq = allgather_vec(var, self.comm)

        if self.comm.rank == 0:
            
            print("Output of 0D vascular model (syspulcapcor_veins):")
            
            print('{:<12s}{:<3s}{:<10.1f}{:<3s}{:<9s}{:<3s}{:<10.1f}'.format(''+self.cname_prfx[2]+'_at_l',' = ',aux[0],'   ',''+self.cname_prfx[3]+'_at_r',' = ',aux[25]))
            print('{:<12s}{:<3s}{:<10.1f}{:<3s}{:<9s}{:<3s}{:<10.1f}'.format(''+self.cname_prfx[0]+'_v_l',' = ',aux[2],'   ',''+self.cname_prfx[1]+'_v_r',' = ',aux[27]))
            
            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format(''+self.vname_prfx[2]+'_at_l',' = ',var_sq[1-self.si[2]],'   ',''+self.vname_prfx[3]+'_at_r',' = ',var_sq[26-self.si[3]]))
            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format(''+self.vname_prfx[0]+'_v_l',' = ',var_sq[3-self.si[0]],'   ',''+self.vname_prfx[1]+'_v_r',' = ',var_sq[28-self.si[1]]))
            
            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format('p_ar_sys',' = ',var_sq[4],'   ','p_ar_pul',' = ',var_sq[29]))
            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format('p_arperi_sys',' = ',var_sq[7],'   ','p_cap_pul',' = ',var_sq[31]))
            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format('p_ven_sys',' = ',var_sq[22],'   ','p_ven_pul',' = ',var_sq[33]))

            sys.stdout.flush()




def postprocess_groups_syspulcapcor_veins(groups, indpertaftercyl=0, multiscalegandr=False):
    
    # index 0
    groups.append({'pres_time_sys_l'  : ['p_at_l', 'p_v_l', 'p_ar_sys', 'p_arperi_sys', 'p_venspl_sys', 'p_venespl_sys', 'p_venmsc_sys', 'p_vencer_sys', 'p_vencor_sys', 'p_ven_sys'],
                'tex'                 : ['$p_{\\\mathrm{at}}^{\\\ell}$', '$p_{\\\mathrm{v}}^{\\\ell}$', '$p_{\\\mathrm{ar}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ar,peri}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ven,spl}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ven,espl}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ven,msc}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ven,cer}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ven,cor}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ven}}^{\\\mathrm{sys}}$'],
                'lines'               : [1, 2, 3, 4, 10, 11, 12, 13, 14, 15]})
    # index 1
    groups.append({'pres_time_pul_r'  : ['p_at_r', 'p_v_r', 'p_ar_pul', 'p_cap_pul', 'p_ven_pul'],
                'tex'                 : ['$p_{\\\mathrm{at}}^{r}$', '$p_{\\\mathrm{v}}^{r}$', '$p_{\\\mathrm{ar}}^{\\\mathrm{pul}}$', '$p_{\\\mathrm{cap}}^{\\\mathrm{pul}}$', '$p_{\\\mathrm{ven}}^{\\\mathrm{pul}}$'],
                'lines'               : [16, 17, 18, 19, 20]})
    # index 2
    groups.append({'flux_time_sys_l'  : ['q_vin_l', 'q_vout_l', 'q_ar_sys', 'q_arspl_sys', 'q_arespl_sys', 'q_armsc_sys', 'q_arcer_sys', 'q_arcor_sys', 'q_venspl_sys', 'q_venespl_sys', 'q_venmsc_sys', 'q_vencer_sys', 'q_vencor_sys', 'q_ven1_sys', 'q_ven2_sys'],
                'tex'                 : ['$q_{\\\mathrm{v,in}}^{\\\ell}$', '$q_{\\\mathrm{v,out}}^{\\\ell}$', '$q_{\\\mathrm{ar}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ar,spl}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ar,espl}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ar,msc}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ar,cer}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ar,cor}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ven,spl}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ven,espl}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ven,msc}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ven,cer}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ven,cor}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ven,1}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ven,2}}^{\\\mathrm{sys}}$'],
                'lines'               : [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15]})
    # index 3
    groups.append({'flux_time_pul_r'  : ['q_vin_r', 'q_vout_r', 'q_ar_pul', 'q_cap_pul', 'q_ven1_pul', 'q_ven2_pul', 'q_ven3_pul', 'q_ven4_pul'],
                'tex'                 : ['$q_{\\\mathrm{v,in}}^{r}$', '$q_{\\\mathrm{v,out}}^{r}$', '$q_{\\\mathrm{ar}}^{\\\mathrm{pul}}$', '$q_{\\\mathrm{cap}}^{\\\mathrm{pul}}$', '$q_{\\\mathrm{ven,1}}^{\\\mathrm{pul}}$', '$q_{\\\mathrm{ven,2}}^{\\\mathrm{pul}}$', '$q_{\\\mathrm{ven,3}}^{\\\mathrm{pul}}$', '$q_{\\\mathrm{ven,4}}^{\\\mathrm{pul}}$'],
                'lines'               : [16, 17, 18, 19, 20, 20, 20, 20]})
    # index 4
    groups.append({'vol_time_l_r'     : ['V_at_l', 'V_v_l', 'V_at_r', 'V_v_r'],
                'tex'                 : ['$V_{\\\mathrm{at}}^{\\\ell}$', '$V_{\\\mathrm{v}}^{\\\ell}$', '$V_{\\\mathrm{at}}^{r}$', '$V_{\\\mathrm{v}}^{r}$'],
                'lines'               : [1, 2, 16, 17]})
    # index 5
    groups.append({'vol_time_compart' : ['V_at_l', 'V_v_l', 'V_at_r', 'V_v_r', 'V_ar_arcor_sys', 'V_arperi_sys', 'V_venspl_sys', 'V_venespl_sys', 'V_venmsc_sys', 'V_vencer_sys', 'V_vencor_sys', 'V_ven_sys', 'V_ar_pul', 'V_cap_pul', 'V_ven_pul', 'V_all'],
                'tex'                 : ['$V_{\\\mathrm{at}}^{\\\ell}$', '$V_{\\\mathrm{v}}^{\\\ell}$', '$V_{\\\mathrm{at}}^{r}$', '$V_{\\\mathrm{v}}^{r}$', '$V_{\\\mathrm{ar}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ar,peri}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven,spl}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven,espl}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven,msc}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven,cer}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven,cor}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ar}}^{\\\mathrm{pul}}$', '$V_{\\\mathrm{cap}}^{\\\mathrm{pul}}$', '$V_{\\\mathrm{ven}}^{\\\mathrm{pul}}$', '$\\\sum V$'],
                'lines'               : [1, 2, 16, 17, 3, 4, 10, 11, 12, 13, 14, 15, 18, 19, 20, 99]})
    
    # pv loops are only considered for the last cycle
    
    if indpertaftercyl > 0: # for comparison of healthy/baseline and perturbed states
        if multiscalegandr:
            # index 6
            groups.append({'pres_vol_v_l_r_PERIODIC'  : ['pV_v_l_gandr', 'pV_v_r_gandr', 'pV_v_l_last', 'pV_v_r_last', 'pV_v_l_baseline', 'pV_v_r_baseline'],
                        'tex'                      : ['$p_{\\\mathrm{v}}^{\\\ell,\\\mathrm{G\\&R}}$', '$p_{\\\mathrm{v}}^{r,\\\mathrm{G\\&R}}$', '$p_{\\\mathrm{v}}^{\\\ell}$', '$p_{\\\mathrm{v}}^{r}$', '$p_{\\\mathrm{v}}^{\\\ell,\\\mathrm{ref}}$', '$p_{\\\mathrm{v}}^{r,\\\mathrm{ref}}$'],
                        'lines'                    : [21, 22, 102, 117, 97, 98]})
            # index 7
            groups.append({'pres_vol_at_l_r_PERIODIC' : ['pV_at_l_gandr', 'pV_at_r_gandr', 'pV_at_l_last', 'pV_at_r_last', 'pV_at_l_baseline', 'pV_at_r_baseline'],
                        'tex'                      : ['$p_{\\\mathrm{at}}^{\\\ell,\\\mathrm{G\\&R}}$', '$p_{\\\mathrm{at}}^{r,\\\mathrm{G\\&R}}$', '$p_{\\\mathrm{at}}^{\\\ell}$', '$p_{\\\mathrm{at}}^{r}$', '$p_{\\\mathrm{at}}^{\\\ell,\\\mathrm{ref}}$', '$p_{\\\mathrm{at}}^{r,\\\mathrm{ref}}$'],
                        'lines'                    : [23, 24, 101, 116, 97, 98]})
        else:
            # index 6
            groups.append({'pres_vol_v_l_r_PERIODIC'  : ['pV_v_l_last', 'pV_v_r_last', 'pV_v_l_baseline', 'pV_v_r_baseline'],
                        'tex'                      : ['$p_{\\\mathrm{v}}^{\\\ell}$', '$p_{\\\mathrm{v}}^{r}$', '$p_{\\\mathrm{v}}^{\\\ell,\\\mathrm{ref}}$', '$p_{\\\mathrm{v}}^{r,\\\mathrm{ref}}$'],
                        'lines'                    : [2, 17, 97, 98]})
            # index 7
            groups.append({'pres_vol_at_l_r_PERIODIC' : ['pV_at_l_last', 'pV_at_r_last', 'pV_at_l_baseline', 'pV_at_r_baseline'],
                        'tex'                      : ['$p_{\\\mathrm{at}}^{\\\ell}$', '$p_{\\\mathrm{at}}^{r}$', '$p_{\\\mathrm{at}}^{\\\ell,\\\mathrm{ref}}$', '$p_{\\\mathrm{at}}^{r,\\\mathrm{ref}}$'],
                        'lines'                    : [1, 16, 97, 98]})
    else:
        # index 6
        groups.append({'pres_vol_v_l_r_PERIODIC'  : ['pV_v_l_last', 'pV_v_r_last'],
                       'tex'                      : ['$p_{\\\mathrm{v}}^{\\\ell}$', '$p_{\\\mathrm{v}}^{r}$'],
                       'lines'                    : [2, 17]})
        # index 7
        groups.append({'pres_vol_at_l_r_PERIODIC' : ['pV_at_l_last', 'pV_at_r_last'],
                       'tex'                      : ['$p_{\\\mathrm{at}}^{\\\ell}$', '$p_{\\\mathrm{at}}^{r}$'],
                       'lines'                    : [1, 16]})
    
    
    # now append all the values again but with suffix PERIODIC, since we want to plot both:
    # values over all heart cycles as well as only for the periodic cycle

    # index 8
    groups.append({'pres_time_sys_l_PERIODIC'  : list(groups[0].values())[0],
                   'tex'                       : list(groups[0].values())[1],
                   'lines'                     : list(groups[0].values())[2]})
    # index 9
    groups.append({'pres_time_pul_r_PERIODIC'  : list(groups[1].values())[0],
                   'tex'                       : list(groups[1].values())[1],
                   'lines'                     : list(groups[1].values())[2]})
    # index 10
    groups.append({'flux_time_sys_l_PERIODIC'  : list(groups[2].values())[0],
                   'tex'                       : list(groups[2].values())[1],
                   'lines'                     : list(groups[2].values())[2]})
    # index 11    
    groups.append({'flux_time_pul_r_PERIODIC'  : list(groups[3].values())[0],
                   'tex'                       : list(groups[3].values())[1],
                   'lines'                     : list(groups[3].values())[2]})
    # index 12        
    groups.append({'vol_time_l_r_PERIODIC'     : list(groups[4].values())[0],
                   'tex'                       : list(groups[4].values())[1],
                   'lines'                     : list(groups[4].values())[2]})
    # index 13            
    groups.append({'vol_time_compart_PERIODIC' : list(groups[5].values())[0],
                   'tex'                       : list(groups[5].values())[1],
                   'lines'                     : list(groups[5].values())[2]})
    
