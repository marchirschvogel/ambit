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

# systemic and pulmonary closed-loop circulation model, each heart chamber can be treated individually,
# either as 0D elastance model, volume or flux coming from a 3D solid, or interface fluxes from a 3D fluid model

# 16 governing equations (uncomment and paste directly into a LaTeX environment):

## left heart and systemic circulation:
#\begin{align}
#&-Q_{\mathrm{at}}^{\ell} = q_{\mathrm{ven}}^{\mathrm{pul}} - q_{\mathrm{v,in}}^{\ell}\nonumber\\
#&\tilde{R}_{\mathrm{v,in}}^{\ell} q_{\mathrm{v,in}}^{\ell} = p_{\mathrm{at}}^{\ell}-p_{\mathrm{v}}^{\ell}\nonumber\\
#&-Q_{\mathrm{v}}^{\ell} = q_{\mathrm{v,in}}^{\ell} - q_{\mathrm{v,out}}^{\ell}\nonumber\\
#&\tilde{R}_{\mathrm{v,out}}^{\ell} q_{\mathrm{v,out}}^{\ell} = p_{\mathrm{v}}^{\ell}-p_{\mathrm{ar}}^{\mathrm{sys}}\nonumber\\
#&C_{\mathrm{ar}}^{\mathrm{sys}} \left(\frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t} - Z_{\mathrm{ar}}^{\mathrm{sys}}\frac{\mathrm{d}q_{\mathrm{v,out}}^{\ell}}{\mathrm{d}t}\right) = q_{\mathrm{v,out}}^{\ell} - q_{\mathrm{ar}}^{\mathrm{sys}}\nonumber\\
#&L_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{sys}}q_{\mathrm{ar}}^{\mathrm{sys}}=p_{\mathrm{ar}}^{\mathrm{sys}} - Z_{\mathrm{ar}}^{\mathrm{sys}}q_{\mathrm{v,out}}^{\ell}-p_{\mathrm{ven}}^{\mathrm{sys}}\nonumber\\
#&C_{\mathrm{ven}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{sys}}-q_{\mathrm{ven}}^{\mathrm{sys}}\nonumber\\
#&L_{\mathrm{ven}}^{\mathrm{sys}} \frac{\mathrm{d}q_{\mathrm{ven}}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ven}}^{\mathrm{sys}} q_{\mathrm{ven}}^{\mathrm{sys}} = p_{\mathrm{ven}}^{\mathrm{sys}} - p_{\mathrm{at}}^{r}\nonumber
#\end{align}

## right heart and pulmonary circulation:
#\begin{align}
#&-Q_{\mathrm{at}}^{r} = q_{\mathrm{ven}}^{\mathrm{sys}} - q_{\mathrm{v,in}}^{r}\nonumber\\
#&\tilde{R}_{\mathrm{v,in}}^{r}q_{\mathrm{v,in}}^{r} = p_{\mathrm{at}}^{r}-p_{\mathrm{v}}^{r}\nonumber\\
#&-Q_{\mathrm{v}}^{r} = q_{\mathrm{v,in}}^{r} - q_{\mathrm{v,out}}^{r}\nonumber\\
#&\tilde{R}_{\mathrm{v,out}}^{r} q_{\mathrm{v,out}}^{r} = p_{\mathrm{v}}^{r}-p_{\mathrm{ar}}^{\mathrm{pul}}\nonumber\\
#&C_{\mathrm{ar}}^{\mathrm{pul}} \left(\frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} - Z_{\mathrm{ar}}^{\mathrm{pul}}\frac{\mathrm{d}q_{\mathrm{v,out}}^{r}}{\mathrm{d}t}\right) = q_{\mathrm{v,out}}^{r} - q_{\mathrm{ar}}^{\mathrm{pul}}\nonumber\\
#&L_{\mathrm{ar}}^{\mathrm{pul}} \frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{pul}} q_{\mathrm{ar}}^{\mathrm{pul}}=p_{\mathrm{ar}}^{\mathrm{pul}} - Z_{\mathrm{ar}}^{\mathrm{pul}}q_{\mathrm{v,out}}^{r}-p_{\mathrm{ven}}^{\mathrm{pul}}\nonumber\\
#&C_{\mathrm{ven}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{pul}} - q_{\mathrm{ven}}^{\mathrm{pul}}\nonumber\\
#&L_{\mathrm{ven}}^{\mathrm{pul}} \frac{\mathrm{d}q_{\mathrm{ven}}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ven}}^{\mathrm{pul}} q_{\mathrm{ven}}^{\mathrm{pul}}=p_{\mathrm{ven}}^{\mathrm{pul}}-p_{\mathrm{at}}^{\ell}\nonumber
#\end{align}

class cardiovascular0Dsyspul(cardiovascular0Dbase):

    def __init__(self, params, chmodels, cq=['volume','volume','volume','volume'], valvelaws={'av' : ['pwlin_pres',0], 'mv' : ['pwlin_pres',0], 'pv' : ['pwlin_pres',0], 'tv' : ['pwlin_pres',0]}, comm=None):
        # initialize base class
        cardiovascular0Dbase.__init__(self, comm=comm)

        # parameters
        # circulatory system parameters: resistances (R), compliances (C), inertances (L), impedances (Z)
        self.R_ar_sys = params['R_ar_sys']
        self.C_ar_sys = params['C_ar_sys']
        self.L_ar_sys = params['L_ar_sys']
        self.Z_ar_sys = params['Z_ar_sys']
        self.R_ven_sys = params['R_ven_sys']
        self.C_ven_sys = params['C_ven_sys']
        self.L_ven_sys = params['L_ven_sys']
        self.R_ar_pul = params['R_ar_pul']
        self.C_ar_pul = params['C_ar_pul']
        self.L_ar_pul = params['L_ar_pul']
        self.Z_ar_pul = params['Z_ar_pul']
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
        
        # valve inertances
        try: self.L_vin_l = params['L_vin_l']
        except: self.L_vin_l = 0
        try: self.L_vin_r = params['L_vin_r']
        except: self.L_vin_r = 0
        try: self.L_vout_l = params['L_vout_l']
        except: self.L_vout_l = 0
        try: self.L_vout_r = params['L_vout_r']
        except: self.L_vout_r = 0
        
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
        self.V_ar_pul_u = params['V_ar_pul_u']
        self.V_ven_sys_u = params['V_ven_sys_u']
        self.V_ven_pul_u = params['V_ven_pul_u']
        
        self.chmodels = chmodels
        self.valvelaws = valvelaws
        
        # number of systemic venous inflows (to right atrium)
        try: self.vs = self.chmodels['ra']['num_inflows']
        except: self.vs = 1
    
        # number of pulmonary venous inflows (to left atrium)
        try: self.vp = self.chmodels['la']['num_inflows']
        except: self.vp = 1
        
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
        self.numdof = 14 + self.vs + self.vp
        
        self.elastarrays = [[]]*4
        
        self.si, self.switch_V, self.switch_p = [0]*4, [1]*4, [0]*4 # default values

        self.vindex_ch = [3,10+self.vs,1,8+self.vs] # coupling variable indices (decreased by 1 for pressure coupling!)
        self.vname_prfx, self.cname = ['p']*4, []

        # set those ids which are relevant for monolithic direct coupling
        self.v_ids, self.c_ids = [], []
        self.cindex_ch = [2,9+self.vs,0,7+self.vs]

        self.set_solve_arrays()


    def evaluate(self, x, t, df=None, f=None, dK=None, K=None, c=[], y=[], a=None):
        
        fnc = self.evaluate_chamber_state(y, t)

        cardiovascular0Dbase.evaluate(self, x, t, df, f, dK, K, c, y, a, fnc)


    def equation_map(self):
        
        self.varmap={'q_vin_l' : 0+self.si[2], ''+self.vname_prfx[2]+'_at_l' : 1-self.si[2], 'q_vout_l' : 2+self.si[0], ''+self.vname_prfx[0]+'_v_l' : 3-self.si[0], 'p_ar_sys' : 4, 'q_ar_sys' : 5, 'p_ven_sys' : 6, 'q_vin_r' : 7+self.vs+self.si[3], ''+self.vname_prfx[3]+'_at_r' : 8+self.vs-self.si[3], 'q_vout_r' : 9+self.vs+self.si[1], ''+self.vname_prfx[1]+'_v_r' : 10+self.vs-self.si[1], 'p_ar_pul' : 11+self.vs, 'q_ar_pul' : 12+self.vs, 'p_ven_pul' : 13+self.vs}

        for n in range(self.vs): self.varmap['q_ven'+str(n+1)+'_sys'] = 7+n
        for n in range(self.vp): self.varmap['q_ven'+str(n+1)+'_pul'] = 14+self.vs+n

        q_ven_sys_, q_ven_pul_ = [], []
        p_at_l_i_, p_at_r_i_ = [], []

        self.t_            = sp.Symbol('t_')
        q_vin_l_           = sp.Symbol('q_vin_l_')
        for n in range(self.vp): p_at_l_i_.append(sp.Symbol('p_at_l_i'+str(n+1)+'_'))
        p_at_l_o1_         = sp.Symbol('p_at_l_o1_')
        q_vout_l_          = sp.Symbol('q_vout_l_')
        p_v_l_i1_, p_v_l_o1_ = sp.Symbol('p_v_l_i1_'), sp.Symbol('p_v_l_o1_')
        p_ar_sys_          = sp.Symbol('p_ar_sys_')
        q_ar_sys_          = sp.Symbol('q_ar_sys_')
        p_ven_sys_         = sp.Symbol('p_ven_sys_')
        for n in range(self.vs): q_ven_sys_.append(sp.Symbol('q_ven'+str(n+1)+'_sys_'))
        q_vin_r_           = sp.Symbol('q_vin_r_')
        for n in range(self.vs): p_at_r_i_.append(sp.Symbol('p_at_r_i'+str(n+1)+'_'))
        p_at_r_o1_         = sp.Symbol('p_at_r_o1_')
        q_vout_r_          = sp.Symbol('q_vout_r_')
        p_v_r_i1_, p_v_r_o1_ = sp.Symbol('p_v_r_i1_'), sp.Symbol('p_v_r_o1_')
        p_ar_pul_          = sp.Symbol('p_ar_pul_')
        q_ar_pul_          = sp.Symbol('q_ar_pul_')
        p_ven_pul_         = sp.Symbol('p_ven_pul_')
        for n in range(self.vp): q_ven_pul_.append(sp.Symbol('q_ven'+str(n+1)+'_pul_'))
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
        self.x_[1-self.si[2]] = p_at_l_i_[0]
        self.x_[2+self.si[0]] = q_vout_l_
        self.x_[3-self.si[0]] = p_v_l_i1_
        self.x_[4] = p_ar_sys_
        self.x_[5] = q_ar_sys_
        self.x_[6] = p_ven_sys_
        for n in range(self.vs):
            self.x_[7+n] = q_ven_sys_[n]
        self.x_[7+self.vs+self.si[3]] = q_vin_r_
        self.x_[8+self.vs-self.si[3]] = p_at_r_i_[0]
        self.x_[9+self.vs+self.si[1]] = q_vout_r_
        self.x_[10+self.vs-self.si[1]] = p_v_r_i1_
        self.x_[11+self.vs] = p_ar_pul_
        self.x_[12+self.vs] = q_ar_pul_
        self.x_[13+self.vs] = p_ven_pul_
        for n in range(self.vp):
            self.x_[14+self.vs+n] = q_ven_pul_[n]

        # set chamber dicts
        chdict_lv = {'vq' : VQ_v_l_, 'pi1' : p_v_l_i1_, 'po1' : p_v_l_o1_}
        chdict_rv = {'vq' : VQ_v_r_, 'pi1' : p_v_r_i1_, 'po1' : p_v_r_o1_}
        chdict_la = {'vq' : VQ_at_l_, 'po1' : p_at_l_o1_}
        for n in range(self.vp): chdict_la['pi'+str(n+1)+''] = p_at_l_i_[n]
        chdict_ra = {'vq' : VQ_at_r_, 'po1' : p_at_r_o1_}
        for n in range(self.vs): chdict_ra['pi'+str(n+1)+''] = p_at_r_i_[n]

        # set chamber states and variables (e.g., express V in terms of p and E in case of elastance models, ...)
        self.set_chamber_state('lv', chdict_lv, [E_v_l_])
        self.set_chamber_state('rv', chdict_rv, [E_v_r_])
        self.set_chamber_state('la', chdict_la, [E_at_l_])
        self.set_chamber_state('ra', chdict_ra, [E_at_r_])

        # feed back modified dicts to chamber variables
        VQ_v_l_, p_v_l_i1_, p_v_l_o1_ = chdict_lv['vq'], chdict_lv['pi1'], chdict_lv['po1']
        VQ_v_r_, p_v_r_i1_, p_v_r_o1_ = chdict_rv['vq'], chdict_rv['pi1'], chdict_rv['po1']
        VQ_at_l_, p_ati1_l_, p_at_l_o1_ = chdict_la['vq'], chdict_la['pi1'], chdict_la['po1']
        for n in range(self.vp): p_at_l_i_[n] = chdict_la['pi'+str(n+1)+'']
        VQ_at_r_, p_ati1_r_, p_at_r_o1_ = chdict_ra['vq'], chdict_ra['pi1'], chdict_ra['po1']
        for n in range(self.vs): p_at_r_i_[n] = chdict_ra['pi'+str(n+1)+'']

       
        # set valve laws - resistive part of q(p) relationship of momentum equation
        vl_mv_, R_vin_l_  = self.valvelaw(p_at_l_o1_,p_v_l_i1_,self.R_vin_l_min,self.R_vin_l_max,self.valvelaws['mv'],self.t_es,self.t_ed)
        vl_av_, R_vout_l_ = self.valvelaw(p_v_l_o1_,p_ar_sys_,self.R_vout_l_min,self.R_vout_l_max,self.valvelaws['av'],self.t_ed,self.t_es)
        vl_tv_, R_vin_r_  = self.valvelaw(p_at_r_o1_,p_v_r_i1_,self.R_vin_r_min,self.R_vin_r_max,self.valvelaws['tv'],self.t_es,self.t_ed)
        vl_pv_, R_vout_r_ = self.valvelaw(p_v_r_o1_,p_ar_pul_,self.R_vout_r_min,self.R_vout_r_max,self.valvelaws['pv'],self.t_ed,self.t_es)

        # parallel venous resistances and inertances:
        # assume that the total venous resistance/inertance distributes equally over all systemic / pulmonary veins that enter the right / left atrium
        # resistance/inertance in parallel: 1/R_total = 1/R_1 + 1/R_2 + ... + 1/R_n, 1/L_total = 1/L_1 + 1/L_2 + ... + 1/L_n
        # let's say: R_1 = R_2 = ... = R_n, L_1 = L_2 = ... = L_n
        
        R_ven_sys, L_ven_sys, R_ven_pul, L_ven_pul = [], [], [], []
        for n in range(self.vs):
            R_ven_sys.append(self.vs*self.R_ven_sys)
            L_ven_sys.append(self.vs*self.L_ven_sys)

        for n in range(self.vp):
            R_ven_pul.append(self.vp*self.R_ven_pul)
            L_ven_pul.append(self.vp*self.L_ven_pul)
        
        # df part of rhs contribution (df - df_old)/dt
        self.df_[0] = VQ_at_l_ * self.switch_V[2]
        self.df_[1] = (self.L_vin_l/R_vin_l_) * q_vin_l_
        self.df_[2] = VQ_v_l_ * self.switch_V[0]
        self.df_[3] = (self.L_vout_l/R_vout_l_) * q_vout_l_
        self.df_[4] = self.C_ar_sys * (p_ar_sys_ - self.Z_ar_sys * q_vout_l_)
        self.df_[5] = (self.L_ar_sys/self.R_ar_sys) * q_ar_sys_
        self.df_[6] = self.C_ven_sys * p_ven_sys_
        for n in range(self.vs):
            self.df_[7+n] = (L_ven_sys[n]/R_ven_sys[n]) * q_ven_sys_[n]
                # -----------------------------------------------------------
        self.df_[7+self.vs] = VQ_at_r_ * self.switch_V[3]
        self.df_[8+self.vs] = (self.L_vin_r/R_vin_r_) * q_vin_r_
        self.df_[9+self.vs] = VQ_v_r_ * self.switch_V[1]
        self.df_[10+self.vs] = (self.L_vout_r/R_vout_r_) * q_vout_r_
        self.df_[11+self.vs] = self.C_ar_pul * (p_ar_pul_ - self.Z_ar_pul * q_vout_r_)
        self.df_[12+self.vs] = (self.L_ar_pul/self.R_ar_pul) * q_ar_pul_
        self.df_[13+self.vs] = self.C_ven_pul * p_ven_pul_
        for n in range(self.vp):
            self.df_[14+self.vs+n] = (L_ven_pul[n]/R_ven_pul[n]) * q_ven_pul_[n]


        # f part of rhs contribution theta * f + (1-theta) * f_old
        self.f_[0] = -sum(q_ven_pul_) + q_vin_l_ - (1-self.switch_V[2]) * VQ_at_l_
        self.f_[1] = vl_mv_ + q_vin_l_
        self.f_[2] = -q_vin_l_ + q_vout_l_ - (1-self.switch_V[0]) * VQ_v_l_
        self.f_[3] = vl_av_ + q_vout_l_
        self.f_[4] = -q_vout_l_ + q_ar_sys_
        self.f_[5] = (p_ven_sys_ - p_ar_sys_ + self.Z_ar_sys * q_vout_l_)/self.R_ar_sys + q_ar_sys_
        self.f_[6] = -q_ar_sys_ + sum(q_ven_sys_)
        for n in range(self.vs):
            self.f_[7+n] = (p_at_r_i_[n]-p_ven_sys_)/R_ven_sys[n] + q_ven_sys_[n]
                # -----------------------------------------------------------
        self.f_[7+self.vs] = -sum(q_ven_sys_) + q_vin_r_ - (1-self.switch_V[3]) * VQ_at_r_
        self.f_[8+self.vs] = vl_tv_ + q_vin_r_
        self.f_[9+self.vs] = -q_vin_r_ + q_vout_r_ - (1-self.switch_V[1]) * VQ_v_r_
        self.f_[10+self.vs] = vl_pv_ + q_vout_r_
        self.f_[11+self.vs] = -q_vout_r_ + q_ar_pul_
        self.f_[12+self.vs] = (p_ven_pul_ - p_ar_pul_ + self.Z_ar_pul * q_vout_r_)/self.R_ar_pul + q_ar_pul_
        self.f_[13+self.vs] = -q_ar_pul_ + sum(q_ven_pul_)
        for n in range(self.vp):
            self.f_[14+self.vs+n] = (p_at_l_i_[n]-p_ven_pul_)/R_ven_pul[n] + q_ven_pul_[n]


        # setup auxiliary variable map
        # coupling variables, 0D chamber volumes, compartment volumes, other shady quantities...
        nc = len(self.c_)
        self.auxmap={}
        for i in range(nc): self.auxmap[self.cname[i]] = i
        if self.chmodels['lv']['type']=='0D_elast' or self.chmodels['lv']['type']=='prescribed' or self.chmodels['lv']['type']=='0D_elast_prescr': self.auxmap['V_v_l'] = nc+0
        if self.chmodels['rv']['type']=='0D_elast' or self.chmodels['rv']['type']=='prescribed' or self.chmodels['rv']['type']=='0D_elast_prescr': self.auxmap['V_v_r'] = nc+1
        if self.chmodels['la']['type']=='0D_elast' or self.chmodels['la']['type']=='prescribed' or self.chmodels['la']['type']=='0D_elast_prescr': self.auxmap['V_at_l'] = nc+2
        if self.chmodels['ra']['type']=='0D_elast' or self.chmodels['ra']['type']=='prescribed' or self.chmodels['ra']['type']=='0D_elast_prescr': self.auxmap['V_at_r'] = nc+3
        self.auxmap['V_ar_sys'] = nc+4
        self.auxmap['V_ven_sys'] = nc+5
        self.auxmap['V_ar_pul'] = nc+6
        self.auxmap['V_ven_pul'] = nc+7

        # populate auxiliary variable vector
        for i in range(nc): self.a_[i] = self.c_[i]
        self.a_[nc+0] = VQ_v_l_ * self.switch_V[0]
        self.a_[nc+1] = VQ_v_r_ * self.switch_V[1]
        self.a_[nc+2] = VQ_at_l_ * self.switch_V[2]
        self.a_[nc+3] = VQ_at_r_ * self.switch_V[3]
        self.a_[nc+4] = self.C_ar_sys * (p_ar_sys_ - self.Z_ar_sys * q_vout_l_) + self.V_ar_sys_u
        self.a_[nc+5] = self.C_ven_sys * p_ven_sys_ + self.V_ven_sys_u
        self.a_[nc+6] = self.C_ar_pul * (p_ar_pul_ - self.Z_ar_pul * q_vout_r_) + self.V_ar_pul_u
        self.a_[nc+7] = self.C_ven_pul * p_ven_pul_ + self.V_ven_pul_u



    def initialize(self, var, iniparam):
        
        var[0+self.si[2]] = iniparam['q_vin_l_0']
        var[1-self.si[2]] = iniparam[''+self.vname_prfx[2]+'_at_l_0']
        var[2+self.si[0]] = iniparam['q_vout_l_0']
        var[3-self.si[0]] = iniparam[''+self.vname_prfx[0]+'_v_l_0']
        var[4] = iniparam['p_ar_sys_0']
        var[5] = iniparam['q_ar_sys_0']
        var[6] = iniparam['p_ven_sys_0']
        for n in range(self.vs):
            try: var[7+n] = iniparam['q_ven'+str(n+1)+'_sys_0']
            except: var[7+n] = iniparam['q_ven_sys_0']
        var[7+self.vs+self.si[3]] = iniparam['q_vin_r_0']
        var[8+self.vs-self.si[3]] = iniparam[''+self.vname_prfx[3]+'_at_r_0']
        var[9+self.vs+self.si[1]] = iniparam['q_vout_r_0']
        var[10+self.vs-self.si[1]] = iniparam[''+self.vname_prfx[1]+'_v_r_0']
        var[11+self.vs] = iniparam['p_ar_pul_0']
        var[12+self.vs] = iniparam['q_ar_pul_0']
        var[13+self.vs] = iniparam['p_ven_pul_0']
        for n in range(self.vp):
            try: var[14+self.vs+n] = iniparam['q_ven'+str(n+1)+'_pul_0']
            except: var[14+self.vs+n] = iniparam['q_ven_pul_0']
                


    def check_periodic(self, varTc, varTc_old, eps, check, cyclerr):
        
        if isinstance(varTc, np.ndarray): varTc_sq, varTc_old_sq = varTc, varTc_old
        else: varTc_sq, varTc_old_sq = allgather_vec(varTc, self.comm), allgather_vec(varTc_old, self.comm)

        if check=='allvar':
            
            vals = []
            for i in range(len(varTc_sq)):
                vals.append( math.fabs((varTc_sq[i]-varTc_old_sq[i])/max(1.0,math.fabs(varTc_old_sq[i]))) )

        elif check=='pQvar':
            
            vals = []
            pQvar_ids = [self.varmap[''+self.vname_prfx[2]+'_at_l'],self.varmap[''+self.vname_prfx[0]+'_v_l'],self.varmap['p_ar_sys'],self.varmap['p_ven_sys'],self.varmap[''+self.vname_prfx[3]+'_at_r'],self.varmap[''+self.vname_prfx[1]+'_v_r'],self.varmap['p_ar_pul'],self.varmap['p_ven_pul']]
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

        nc = len(self.c_)

        if self.comm.rank == 0:
            
            print("Output of 0D vascular model (syspul):")

            for i in range(nc):
                print('{:<9s}{:<3s}{:<10.3f}'.format(list(self.auxmap.keys())[i],' = ',aux[list(self.auxmap.values())[i]]))
                        
            print('{:<9s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format(''+self.vname_prfx[2]+'_at_l',' = ',var_sq[self.varmap[''+self.vname_prfx[2]+'_at_l']],'   ',''+self.vname_prfx[3]+'_at_r',' = ',var_sq[self.varmap[''+self.vname_prfx[3]+'_at_r']]))
            print('{:<9s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format(''+self.vname_prfx[0]+'_v_l',' = ',var_sq[self.varmap[''+self.vname_prfx[0]+'_v_l']],'   ',''+self.vname_prfx[1]+'_v_r',' = ',var_sq[self.varmap[''+self.vname_prfx[1]+'_v_r']]))
            
            print('{:<9s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format('p_ar_sys',' = ',var_sq[self.varmap['p_ar_sys']],'   ','p_ar_pul',' = ',var_sq[self.varmap['p_ar_pul']]))
            print('{:<9s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format('p_ven_sys',' = ',var_sq[self.varmap['p_ven_sys']],'   ','p_ven_pul',' = ',var_sq[self.varmap['p_ven_pul']]))

            sys.stdout.flush()
            



def postprocess_groups_syspul(groups, indpertaftercyl=0, multiscalegandr=False):
    
    # index 0
    groups.append({'pres_time_sys_l'  : ['p_at_l', 'p_v_l', 'p_ar_sys', 'p_ven_sys'],
                'tex'              : ['$p_{\\\mathrm{at}}^{\\\ell}$', '$p_{\\\mathrm{v}}^{\\\ell}$', '$p_{\\\mathrm{ar}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ven}}^{\\\mathrm{sys}}$'],
                'lines'            : [1, 2, 3, 15]})
    # index 1
    groups.append({'pres_time_pul_r'  : ['p_at_r', 'p_v_r', 'p_ar_pul', 'p_ven_pul'],
                'tex'              : ['$p_{\\\mathrm{at}}^{r}$', '$p_{\\\mathrm{v}}^{r}$', '$p_{\\\mathrm{ar}}^{\\\mathrm{pul}}$', '$p_{\\\mathrm{ven}}^{\\\mathrm{pul}}$'],
                'lines'            : [16, 17, 18, 20]})
    # index 2
    groups.append({'flux_time_sys_l'  : ['q_vin_l', 'q_vout_l', 'q_ar_sys', 'q_ven1_sys', 'q_ven2_sys'],
                'tex'              : ['$q_{\\\mathrm{v,in}}^{\\\ell}$', '$q_{\\\mathrm{v,out}}^{\\\ell}$', '$q_{\\\mathrm{ar}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ven,1}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ven,2}}^{\\\mathrm{sys}}$'],
                'lines'            : [1, 2, 3, 15, 151]})
    # index 3
    groups.append({'flux_time_pul_r'  : ['q_vin_r', 'q_vout_r', 'q_ar_pul', 'q_ven1_pul', 'q_ven2_pul', 'q_ven3_pul', 'q_ven4_pul'],
                'tex'              : ['$q_{\\\mathrm{v,in}}^{r}$', '$q_{\\\mathrm{v,out}}^{r}$', '$q_{\\\mathrm{ar}}^{\\\mathrm{pul}}$', '$q_{\\\mathrm{ven,1}}^{\\\mathrm{pul}}$', '$q_{\\\mathrm{ven,2}}^{\\\mathrm{pul}}$', '$q_{\\\mathrm{ven,3}}^{\\\mathrm{pul}}$', '$q_{\\\mathrm{ven,4}}^{\\\mathrm{pul}}$'],
                'lines'            : [16, 17, 18, 20, 201, 202, 203]})    
    # index 4
    groups.append({'vol_time_l_r'     : ['V_at_l', 'V_v_l', 'V_at_r', 'V_v_r'],
                'tex'              : ['$V_{\\\mathrm{at}}^{\\\ell}$', '$V_{\\\mathrm{v}}^{\\\ell}$', '$V_{\\\mathrm{at}}^{r}$', '$V_{\\\mathrm{v}}^{r}$'],
                'lines'            : [1, 2, 16, 17]})    
    # index 5
    groups.append({'vol_time_compart' : ['V_at_l', 'V_v_l', 'V_at_r', 'V_v_r', 'V_ar_sys', 'V_ven_sys', 'V_ar_pul', 'V_ven_pul', 'V_all'],
                'tex'              : ['$V_{\\\mathrm{at}}^{\\\ell}$', '$V_{\\\mathrm{v}}^{\\\ell}$', '$V_{\\\mathrm{at}}^{r}$', '$V_{\\\mathrm{v}}^{r}$', '$V_{\\\mathrm{ar}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ar}}^{\\\mathrm{pul}}$', '$V_{\\\mathrm{ven}}^{\\\mathrm{pul}}$', '$\\\sum V$'],
                'lines'            : [1, 2, 16, 17, 3, 15, 18, 20, 99]})
    
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
