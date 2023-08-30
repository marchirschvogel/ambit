#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import sys, math
import numpy as np
import sympy as sp

from .cardiovascular0D import cardiovascular0Dbase
from ..mpiroutines import allgather_vec

# systemic and pulmonary closed-loop circulation model including capillary flow, each heart chamber can be treated individually,
# either as 0D elastance model, volume or flux coming from a 3D solid, or interface fluxes from a 3D fluid model

# 36 governing equations (uncomment and paste directly into a LaTeX environment):

#% left heart and systemic circulation:
#\begin{align}
#&-Q_{\mathrm{at}}^{\ell} = q_{\mathrm{ven}}^{\mathrm{pul}} - q_{\mathrm{v,in}}^{\ell}\nonumber\\
#&\tilde{R}_{\mathrm{v,in}}^{\ell}\,q_{\mathrm{v,in}}^{\ell} = p_{\mathrm{at}}^{\ell}-p_{\mathrm{v}}^{\ell}\nonumber\\
#&-Q_{\mathrm{v}}^{\ell} = q_{\mathrm{v,in}}^{\ell} - q_{\mathrm{v,out}}^{\ell}\nonumber\\
#&\tilde{R}_{\mathrm{v,out}}^{\ell}\,q_{\mathrm{v,out}}^{\ell} = p_{\mathrm{v}}^{\ell}-p_{\mathrm{ar}}^{\mathrm{sys}}\nonumber\\
#&0 = q_{\mathrm{v,out}}^{\ell} - q_{\mathrm{ar,p}}^{\mathrm{sys}}\nonumber\\
#&I_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}q_{\mathrm{ar,p}}^{\mathrm{sys}}}{\mathrm{d}t} + Z_{\mathrm{ar}}^{\mathrm{sys}}\,q_{\mathrm{ar,p}}^{\mathrm{sys}}=p_{\mathrm{ar}}^{\mathrm{sys}}-p_{\mathrm{ar,d}}^{\mathrm{sys}}\nonumber\\
#&C_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ar,d}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar,p}}^{\mathrm{sys}} - q_{\mathrm{ar}}^{\mathrm{sys}}\nonumber\\
#&L_{\mathrm{ar}}^{\mathrm{sys}}\frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{sys}}\,q_{\mathrm{ar}}^{\mathrm{sys}}=p_{\mathrm{ar,d}}^{\mathrm{sys}} -p_{\mathrm{ar,peri}}^{\mathrm{sys}}\nonumber\\
#&\left(\sum_{j\in\{\mathrm{spl,espl,\atop msc,cer,cor}\}}\!\!\!\!\!\!\!\!\!C_{\mathrm{ar},j}^{\mathrm{sys}}\right) \frac{\mathrm{d}p_{\mathrm{ar,peri}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{sys}}-\!\!\!\!\!\sum_{j\in\{\mathrm{spl,espl,\atop msc,cer,cor}\}}\!\!\!\!\!\!\!\!\!q_{\mathrm{ar},j}^{\mathrm{sys}}\nonumber\\
#&R_{\mathrm{ar},i}^{\mathrm{sys}}\,q_{\mathrm{ar},i}^{\mathrm{sys}} = p_{\mathrm{ar,peri}}^{\mathrm{sys}} - p_{\mathrm{ven},i}^{\mathrm{sys}}, \quad\scriptstyle{i\in\{\mathrm{spl,espl,\atop msc,cer,cor}\}}\nonumber\\
#&C_{\mathrm{ven},i}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven},i}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar},i}^{\mathrm{sys}} - q_{\mathrm{ven},i}^{\mathrm{sys}}, \quad\scriptstyle{i\in\{\mathrm{spl,espl,\atop msc,cer,cor}\}}\nonumber\\
#&R_{\mathrm{ven},i}^{\mathrm{sys}}\,q_{\mathrm{ven},i}^{\mathrm{sys}} = p_{\mathrm{ven},i}^{\mathrm{sys}}-p_{\mathrm{ven}}^{\mathrm{sys}}, \quad\scriptstyle{i\in\{\mathrm{spl,espl,\atop msc,cer,cor}\}}\nonumber\\
#&C_{\mathrm{ven}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{sys}}}{\mathrm{d}t} = \!\!\!\!\sum_{j=\mathrm{spl,espl,\atop msc,cer,cor}}\!\!\!\!\!q_{\mathrm{ven},j}^{\mathrm{sys}}-q_{\mathrm{ven}}^{\mathrm{sys}}\nonumber\\
#&L_{\mathrm{ven}}^{\mathrm{sys}}\frac{\mathrm{d}q_{\mathrm{ven}}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ven}}^{\mathrm{sys}}\, q_{\mathrm{ven}}^{\mathrm{sys}} = p_{\mathrm{ven}}^{\mathrm{sys}} - p_{\mathrm{at}}^{r}\nonumber
#\end{align}

#% right heart and pulmonary circulation:
#\begin{align}
#&-Q_{\mathrm{at}}^{r} = q_{\mathrm{ven}}^{\mathrm{sys}} - q_{\mathrm{v,in}}^{r}\nonumber\\
#&\tilde{R}_{\mathrm{v,in}}^{r}\,q_{\mathrm{v,in}}^{r} = p_{\mathrm{at}}^{r}-p_{\mathrm{v}}^{r}\nonumber\\
#&-Q_{\mathrm{v}}^{r} = q_{\mathrm{v,in}}^{r} - q_{\mathrm{v,out}}^{r}\nonumber\\
#&\tilde{R}_{\mathrm{v,out}}^{r}\,q_{\mathrm{v,out}}^{r} = p_{\mathrm{v}}^{r}-p_{\mathrm{ar}}^{\mathrm{pul}}\nonumber\\
#&C_{\mathrm{ar}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{v,out}}^{r} - q_{\mathrm{ar}}^{\mathrm{pul}}\nonumber\\
#&L_{\mathrm{ar}}^{\mathrm{pul}}\frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{pul}}\,q_{\mathrm{ar}}^{\mathrm{pul}}=p_{\mathrm{ar}}^{\mathrm{pul}} -p_{\mathrm{cap}}^{\mathrm{pul}}\nonumber\\
#&C_{\mathrm{cap}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{cap}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{pul}} - q_{\mathrm{cap}}^{\mathrm{pul}}\nonumber\\
#&R_{\mathrm{cap}}^{\mathrm{pul}}\,q_{\mathrm{cap}}^{\mathrm{pul}}=p_{\mathrm{cap}}^{\mathrm{pul}}-p_{\mathrm{ven}}^{\mathrm{pul}}\nonumber\\
#&C_{\mathrm{ven}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{cap}}^{\mathrm{pul}} - q_{\mathrm{ven}}^{\mathrm{pul}}\nonumber\\
#&L_{\mathrm{ven}}^{\mathrm{pul}}\frac{\mathrm{d}q_{\mathrm{ven}}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ven}}^{\mathrm{pul}}\, q_{\mathrm{ven}}^{\mathrm{pul}}=p_{\mathrm{ven}}^{\mathrm{pul}}-p_{\mathrm{at}}^{\ell}\nonumber
#\end{align}

class cardiovascular0Dsyspulcap(cardiovascular0Dbase):

    def __init__(self, params, chmodels, cq, vq, valvelaws={'av' : ['pwlin_pres',0], 'mv' : ['pwlin_pres',0], 'pv' : ['pwlin_pres',0], 'tv' : ['pwlin_pres',0]}, cormodel=None, vadmodel=None, init=True, comm=None):
        # initialize base class
        super().__init__(init=init, comm=comm)

        # parameters
        # circulatory system parameters: resistances (R), compliances (C), inertances (L, I), impedances (Z)
        self.R_ar_sys = params['R_ar_sys']
        self.C_ar_sys = params['C_ar_sys']
        self.L_ar_sys = params['L_ar_sys']
        self.Z_ar_sys = params['Z_ar_sys']
        try: self.I_ar_sys = params['I_ar_sys']
        except: self.I_ar_sys = 0

        # peripheral arterial compliances and resistances
        self.R_arspl_sys = params['R_arspl_sys']
        self.C_arspl_sys = params['C_arspl_sys']
        self.R_arespl_sys = params['R_arespl_sys']
        self.C_arespl_sys = params['C_arespl_sys']
        self.R_armsc_sys = params['R_armsc_sys']
        self.C_armsc_sys = params['C_armsc_sys']
        self.R_arcer_sys = params['R_arcer_sys']
        self.C_arcer_sys = params['C_arcer_sys']
        try: self.R_arcor_sys = params['R_arcor_sys']
        except: self.R_arcor_sys = params['R_corp_sys']
        try: self.C_arcor_sys = params['C_arcor_sys']
        except: self.C_arcor_sys = params['C_corp_sys']
        # peripheral venous compliances and resistances
        self.R_venspl_sys = params['R_venspl_sys']
        self.C_venspl_sys = params['C_venspl_sys']
        self.R_venespl_sys = params['R_venespl_sys']
        self.C_venespl_sys = params['C_venespl_sys']
        self.R_venmsc_sys = params['R_venmsc_sys']
        self.C_venmsc_sys = params['C_venmsc_sys']
        self.R_vencer_sys = params['R_vencer_sys']
        self.C_vencer_sys = params['C_vencer_sys']
        try: self.R_vencor_sys = params['R_vencor_sys']
        except: self.R_vencor_sys = params['R_cord_sys']
        try: self.C_vencor_sys = params['C_vencor_sys']
        except: self.C_vencor_sys = params['C_cord_sys']

        self.R_ar_pul = params['R_ar_pul']
        self.C_ar_pul = params['C_ar_pul']
        self.L_ar_pul = params['L_ar_pul']
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
        try: self.V_at_l_u = params['V_at_l_u']
        except: self.V_at_l_u = 0
        try: self.V_at_r_u = params['V_at_r_u']
        except: self.V_at_r_u = 0
        try: self.V_v_l_u = params['V_v_l_u']
        except: self.V_v_l_u = 0
        try: self.V_v_r_u = params['V_v_r_u']
        except: self.V_v_r_u = 0
        try: self.V_ar_sys_u = params['V_ar_sys_u']
        except: self.V_ar_sys_u = 0
        try: self.V_arspl_sys_u = params['V_arspl_sys_u']
        except: self.V_arspl_sys_u = 0
        try: self.V_arespl_sys_u = params['V_arespl_sys_u']
        except: self.V_arespl_sys_u = 0
        try: self.V_armsc_sys_u = params['V_armsc_sys_u']
        except: self.V_armsc_sys_u = 0
        try: self.V_arcer_sys_u = params['V_arcer_sys_u']
        except: self.V_arcer_sys_u = 0
        try: self.V_arcor_sys_u = params['V_arcor_sys_u']
        except: self.V_arcor_sys_u = 0
        try: self.V_venspl_sys_u = params['V_venspl_sys_u']
        except: self.V_venspl_sys_u = 0
        try: self.V_venespl_sys_u = params['V_venespl_sys_u']
        except: self.V_venespl_sys_u = 0
        try: self.V_venmsc_sys_u = params['V_venmsc_sys_u']
        except: self.V_venmsc_sys_u = 0
        try: self.V_vencer_sys_u = params['V_vencer_sys_u']
        except: self.V_vencer_sys_u = 0
        try: self.V_vencor_sys_u = params['V_vencor_sys_u']
        except: self.V_vencor_sys_u = 0
        try: self.V_ven_sys_u = params['V_ven_sys_u']
        except: self.V_ven_sys_u = 0
        try: self.V_ar_pul_u = params['V_ar_pul_u']
        except: self.V_ar_pul_u = 0
        try: self.V_cap_pul_u = params['V_cap_pul_u']
        except: self.V_cap_pul_u = 0
        try: self.V_ven_pul_u = params['V_ven_pul_u']
        except: self.V_ven_pul_u = 0

        self.params = params

        self.chmodels = chmodels
        self.cormodel = cormodel
        self.valvelaws = valvelaws

        # number of systemic venous inflows (to right atrium)
        try: self.vs = self.chmodels['ra']['num_inflows']
        except: self.vs = 1

        # number of pulmonary venous inflows (to left atrium)
        try: self.vp = self.chmodels['la']['num_inflows']
        except: self.vp = 1

        self.cq = cq
        self.vq = vq

        # set up arrays
        self.setup_arrays()

        # setup compartments
        self.set_compartment_interfaces()

        # set up symbolic equations
        self.equation_map()

        # symbolic stiffness matrix
        self.set_stiffness()

        # make Lambda functions out of symbolic expressions
        self.lambdify_expressions()


    def setup_arrays(self):

        # number of degrees of freedom
        self.numdof = 34 + self.vs + self.vp

        self.elastarrays = [[]]*4

        self.si, self.switch_V = [0]*5, [1]*5 # default values

        self.vindex_ch = [3,28+self.vs,1,26+self.vs] # coupling variable indices (decreased by 1 for pressure coupling!)
        self.vname, self.cname = ['p_v_l','p_v_r','p_at_l','p_at_r'], []

        # set those ids which are relevant for monolithic direct coupling
        self.v_ids, self.c_ids = [], []
        self.cindex_ch = [2,27+self.vs,0,25+self.vs]

        self.set_solve_arrays()


    def evaluate(self, x, t, df=None, f=None, dK=None, K=None, c=[], y=[], a=None):

        fnc = self.evaluate_chamber_state(y, t)

        cardiovascular0Dbase.evaluate(self, x, t, df, f, dK, K, c, y, a, fnc)


    def equation_map(self):

        # variable map
        self.varmap['q_vin_l']                   = 0+self.si[2]
        self.varmap[self.vname[2]]               = 1-self.si[2]
        self.varmap['q_vout_l']                  = 2+self.si[0]
        self.varmap[self.vname[0]]               = 3+self.si[0]
        self.varmap['p_ar_sys']                  = 4
        self.varmap['q_arp_sys']                 = 5
        self.varmap['p_ard_sys']                 = 6
        self.varmap['q_ar_sys']                  = 7
        self.varmap['p_arperi_sys']              = 8
        self.varmap['q_arspl_sys']               = 9
        self.varmap['q_arespl_sys']              = 10
        self.varmap['q_armsc_sys']               = 11
        self.varmap['q_arcer_sys']               = 12
        self.varmap['q_arcor_sys']               = 13
        self.varmap['p_venspl_sys']              = 14
        self.varmap['q_venspl_sys']              = 15
        self.varmap['p_venespl_sys']             = 16
        self.varmap['q_venespl_sys']             = 17
        self.varmap['p_venmsc_sys']              = 18
        self.varmap['q_venmsc_sys']              = 19
        self.varmap['p_vencer_sys']              = 20
        self.varmap['q_vencer_sys']              = 21
        self.varmap['p_vencor_sys']              = 22
        self.varmap['q_vencor_sys']              = 23
        self.varmap['p_ven_sys']                 = 24
        for n in range(self.vs):
            self.varmap['q_ven'+str(n+1)+'_sys'] = 25+n
        self.varmap['q_vin_r']                   = 25+self.vs+self.si[3]
        self.varmap[self.vname[3]]               = 26+self.vs-self.si[3]
        self.varmap['q_vout_r']                  = 27+self.vs+self.si[1]
        self.varmap[self.vname[1]]               = 28+self.vs-self.si[1]
        self.varmap['p_ar_pul']                  = 29+self.vs
        self.varmap['q_ar_pul']                  = 30+self.vs
        self.varmap['p_cap_pul']                 = 31+self.vs
        self.varmap['q_cap_pul']                 = 32+self.vs
        self.varmap['p_ven_pul']                 = 33+self.vs
        for n in range(self.vp):
            self.varmap['q_ven'+str(n+1)+'_pul'] = 34+self.vs+n

        q_ven_sys_, q_ven_pul_ = [], []
        p_at_l_i_, p_at_r_i_ = [], []

        self.t_            = sp.Symbol('t_')
        q_vin_l_           = sp.Symbol('q_vin_l_')
        for n in range(self.vp): p_at_l_i_.append(sp.Symbol('p_at_l_i'+str(n+1)+'_'))
        p_at_l_o1_         = sp.Symbol('p_at_l_o1_')
        q_vout_l_          = sp.Symbol('q_vout_l_')
        p_v_l_i1_, p_v_l_o1_ = sp.Symbol('p_v_l_i1_'), sp.Symbol('p_v_l_o1_')
        p_ar_sys_          = sp.Symbol('p_ar_sys_')
        q_arp_sys_      = sp.Symbol('q_arp_sys_')
        p_ard_sys_      = sp.Symbol('p_ard_sys_')
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
        for n in range(self.vs): q_ven_sys_.append(sp.Symbol('q_ven'+str(n+1)+'_sys_'))
        q_vin_r_           = sp.Symbol('q_vin_r_')
        for n in range(self.vs): p_at_r_i_.append(sp.Symbol('p_at_r_i'+str(n+1)+'_'))
        p_at_r_o1_         = sp.Symbol('p_at_r_o1_')
        q_vout_r_          = sp.Symbol('q_vout_r_')
        p_v_r_i1_, p_v_r_o1_ = sp.Symbol('p_v_r_i1_'), sp.Symbol('p_v_r_o1_')
        p_ar_pul_          = sp.Symbol('p_ar_pul_')
        q_ar_pul_          = sp.Symbol('q_ar_pul_')
        p_cap_pul_         = sp.Symbol('p_cap_pul_')
        q_cap_pul_         = sp.Symbol('q_cap_pul_')
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
        self.x_[self.varmap['q_vin_l']]                   = q_vin_l_
        self.x_[self.varmap[self.vname[2]]]               = p_at_l_i_[0]
        self.x_[self.varmap['q_vout_l']]                  = q_vout_l_
        self.x_[self.varmap[self.vname[0]]]               = p_v_l_i1_
        self.x_[self.varmap['p_ar_sys']]                  = p_ar_sys_
        self.x_[self.varmap['q_arp_sys']]                 = q_arp_sys_
        self.x_[self.varmap['p_ard_sys']]                 = p_ard_sys_
        self.x_[self.varmap['q_ar_sys']]                  = q_ar_sys_
        self.x_[self.varmap['p_arperi_sys']]              = p_arperi_sys_
        self.x_[self.varmap['q_arspl_sys']]               = q_arspl_sys_
        self.x_[self.varmap['q_arespl_sys']]              = q_arespl_sys_
        self.x_[self.varmap['q_armsc_sys']]               = q_armsc_sys_
        self.x_[self.varmap['q_arcer_sys']]               = q_arcer_sys_
        self.x_[self.varmap['q_arcor_sys']]               = q_arcor_sys_
        self.x_[self.varmap['p_venspl_sys']]              = p_venspl_sys_
        self.x_[self.varmap['q_venspl_sys']]              = q_venspl_sys_
        self.x_[self.varmap['p_venespl_sys']]             = p_venespl_sys_
        self.x_[self.varmap['q_venespl_sys']]             = q_venespl_sys_
        self.x_[self.varmap['p_venmsc_sys']]              = p_venmsc_sys_
        self.x_[self.varmap['q_venmsc_sys']]              = q_venmsc_sys_
        self.x_[self.varmap['p_vencer_sys']]              = p_vencer_sys_
        self.x_[self.varmap['q_vencer_sys']]              = q_vencer_sys_
        self.x_[self.varmap['p_vencor_sys']]              = p_vencor_sys_
        self.x_[self.varmap['q_vencor_sys']]              = q_vencor_sys_
        self.x_[self.varmap['p_ven_sys']]                 = p_ven_sys_
        for n in range(self.vs):
            self.x_[self.varmap['q_ven'+str(n+1)+'_sys']] = q_ven_sys_[n]
        self.x_[self.varmap['q_vin_r']]                   = q_vin_r_
        self.x_[self.varmap[self.vname[3]]]               = p_at_r_i_[0]
        self.x_[self.varmap['q_vout_r']]                  = q_vout_r_
        self.x_[self.varmap[self.vname[1]]]               = p_v_r_i1_
        self.x_[self.varmap['p_ar_pul']]                  = p_ar_pul_
        self.x_[self.varmap['q_ar_pul']]                  = q_ar_pul_
        self.x_[self.varmap['p_cap_pul']]                 = p_cap_pul_
        self.x_[self.varmap['q_cap_pul']]                 = q_cap_pul_
        self.x_[self.varmap['p_ven_pul']]                 = p_ven_pul_
        for n in range(self.vp):
            self.x_[self.varmap['q_ven'+str(n+1)+'_pul']] = q_ven_pul_[n]

        # set chamber dicts
        chdict_lv = {'VQ' : VQ_v_l_, 'pi1' : p_v_l_i1_, 'po1' : p_v_l_o1_}
        chdict_rv = {'VQ' : VQ_v_r_, 'pi1' : p_v_r_i1_, 'po1' : p_v_r_o1_}
        chdict_la = {'VQ' : VQ_at_l_, 'po1' : p_at_l_o1_}
        for n in range(self.vp): chdict_la['pi'+str(n+1)] = p_at_l_i_[n]
        chdict_ra = {'VQ' : VQ_at_r_, 'po1' : p_at_r_o1_}
        for n in range(self.vs): chdict_ra['pi'+str(n+1)] = p_at_r_i_[n]

        # set coupling states and variables (e.g., express V in terms of p and E in case of elastance models, ...)
        self.set_coupling_state('lv', chdict_lv, [E_v_l_])
        self.set_coupling_state('rv', chdict_rv, [E_v_r_])
        self.set_coupling_state('la', chdict_la, [E_at_l_])
        self.set_coupling_state('ra', chdict_ra, [E_at_r_])

        # feed back modified dicts to chamber variables
        VQ_v_l_, p_v_l_i1_, p_v_l_o1_ = chdict_lv['VQ'], chdict_lv['pi1'], chdict_lv['po1']
        VQ_v_r_, p_v_r_i1_, p_v_r_o1_ = chdict_rv['VQ'], chdict_rv['pi1'], chdict_rv['po1']
        VQ_at_l_, p_ati1_l_, p_at_l_o1_ = chdict_la['VQ'], chdict_la['pi1'], chdict_la['po1']
        for n in range(self.vp): p_at_l_i_[n] = chdict_la['pi'+str(n+1)]
        VQ_at_r_, p_ati1_r_, p_at_r_o1_ = chdict_ra['VQ'], chdict_ra['pi1'], chdict_ra['po1']
        for n in range(self.vs): p_at_r_i_[n] = chdict_ra['pi'+str(n+1)]


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
        self.df_[0]  = VQ_at_l_ * self.switch_V[2]                                                                             # left atrium volume rate
        self.df_[1]  = (self.L_vin_l/R_vin_l_) * q_vin_l_                                                                      # mitral valve inertia
        self.df_[2]  = VQ_v_l_ * self.switch_V[0]                                                                              # left ventricle volume rate
        self.df_[3]  = (self.L_vout_l/R_vout_l_) * q_vout_l_                                                                   # aortic valve inertia
        self.df_[4]  = 0.
        self.df_[5]  = (self.I_ar_sys/self.Z_ar_sys) * q_arp_sys_                                                           # aortic root inertia
        self.df_[6]  = self.C_ar_sys * p_ard_sys_                                                                           # systemic arterial volume rate
        self.df_[7]  = (self.L_ar_sys/self.R_ar_sys) * q_ar_sys_                                                               # systemic arterial inertia
        self.df_[8]  = (self.C_arspl_sys+self.C_arespl_sys+self.C_armsc_sys+self.C_arcer_sys+self.C_arcor_sys) * p_arperi_sys_ # systemic capillary volume rate
        self.df_[9]  = 0.
        self.df_[10] = 0.
        self.df_[11] = 0.
        self.df_[12] = 0.
        self.df_[13] = 0.
        self.df_[14] = self.C_venspl_sys * p_venspl_sys_                                                                       # systemic venous splanchnic volume rate
        self.df_[15] = 0.
        self.df_[16] = self.C_venespl_sys * p_venespl_sys_                                                                     # systemic venous extra-splanchnic volume rate
        self.df_[17] = 0.
        self.df_[18] = self.C_venmsc_sys * p_venmsc_sys_                                                                       # systemic venous muscular volume rate
        self.df_[19] = 0.
        self.df_[20] = self.C_vencer_sys * p_vencer_sys_                                                                       # systemic venous cerebral volume rate
        self.df_[21] = 0.
        self.df_[22] = self.C_vencor_sys * p_vencor_sys_                                                                       # systemic venous coronary volume rate
        self.df_[23] = 0.
        self.df_[24] = self.C_ven_sys * p_ven_sys_                                                                             # systemic venous volume rate
        for n in range(self.vs):
            self.df_[25+n] = (L_ven_sys[n]/R_ven_sys[n]) * q_ven_sys_[n]                                                       # systemic venous inertia
                # -----------------------------------------------------------
        self.df_[25+self.vs] = VQ_at_r_ * self.switch_V[3]                                                                     # right atrium volume rate
        self.df_[26+self.vs] = (self.L_vin_r/R_vin_r_) * q_vin_r_                                                              # tricuspid valve inertia
        self.df_[27+self.vs] = VQ_v_r_ * self.switch_V[1]                                                                      # right ventricle volume rate
        self.df_[28+self.vs] = (self.L_vout_r/R_vout_r_) * q_vout_r_                                                           # pulmonary valve inertia
        self.df_[29+self.vs] = self.C_ar_pul * p_ar_pul_                                                                       # pulmonary arterial volume rate
        self.df_[30+self.vs] = (self.L_ar_pul/self.R_ar_pul) * q_ar_pul_                                                       # pulmonary arterial inertia
        self.df_[31+self.vs] = self.C_cap_pul * p_cap_pul_                                                                     # pulmonary capillary volume rate
        self.df_[32+self.vs] = 0.
        self.df_[33+self.vs] = self.C_ven_pul * p_ven_pul_                                                                     # pulmonary venous volume rate
        for n in range(self.vp):
            self.df_[34+self.vs+n] = (L_ven_pul[n]/R_ven_pul[n]) * q_ven_pul_[n]                                               # pulmonary venous inertia


        # f part of rhs contribution theta * f + (1-theta) * f_old
        self.f_[0]  = -sum(q_ven_pul_) + q_vin_l_ - (1-self.switch_V[2]) * VQ_at_l_                                            # left atrium flow balance
        self.f_[1]  = vl_mv_ + q_vin_l_                                                                                        # mitral valve momentum
        self.f_[2]  = -q_vin_l_ + q_vout_l_ - (1-self.switch_V[0]) * VQ_v_l_                                                   # left ventricle flow balance
        self.f_[3]  = vl_av_ + q_vout_l_                                                                                       # aortic valve momentum
        self.f_[4]  = -q_vout_l_ + q_arp_sys_                                                                               # aortic root flow balance
        self.f_[5]  = (p_ard_sys_ - p_ar_sys_)/self.Z_ar_sys + q_arp_sys_                                                # aortic root momentum
        self.f_[6]  = -q_arp_sys_ + q_ar_sys_                                                                               # systemic arterial flow balance
        self.f_[7]  = (p_arperi_sys_ - p_ard_sys_)/self.R_ar_sys + q_ar_sys_                                                # systemic arterial momentum
        self.f_[8]  = -q_ar_sys_ + (q_arspl_sys_ + q_arespl_sys_ + q_armsc_sys_ + q_arcer_sys_ + q_arcor_sys_)                 # systemic capillary flow balance
        self.f_[9]  = (p_venspl_sys_ - p_arperi_sys_)/self.R_arspl_sys + q_arspl_sys_                                          # systemic arterial splanchnic momentum
        self.f_[10] = (p_venespl_sys_ - p_arperi_sys_)/self.R_arespl_sys + q_arespl_sys_                                       # systemic arterial extra-splanchnic momentum
        self.f_[11] = (p_venmsc_sys_ - p_arperi_sys_)/self.R_armsc_sys + q_armsc_sys_                                          # systemic arterial muscular momentum
        self.f_[12] = (p_vencer_sys_ - p_arperi_sys_)/self.R_arcer_sys + q_arcer_sys_                                          # systemic arterial cerebral momentum
        self.f_[13] = (p_vencor_sys_ - p_arperi_sys_)/self.R_arcor_sys + q_arcor_sys_                                          # systemic arterial coronary momentum
        self.f_[14] = q_venspl_sys_ - q_arspl_sys_                                                                             # systemic venous splanchnic flow balance
        self.f_[15] = (p_ven_sys_ - p_venspl_sys_)/self.R_venspl_sys + q_venspl_sys_                                           # systemic venous splanchnic momentum
        self.f_[16] = q_venespl_sys_ - q_arespl_sys_                                                                           # systemic venous extra-splanchnic flow balance
        self.f_[17] = (p_ven_sys_ - p_venespl_sys_)/self.R_venespl_sys + q_venespl_sys_                                        # systemic venous extra-splanchnic momentum
        self.f_[18] = q_venmsc_sys_ - q_armsc_sys_                                                                             # systemic venous muscular flow balance
        self.f_[19] = (p_ven_sys_ - p_venmsc_sys_)/self.R_venmsc_sys + q_venmsc_sys_                                           # systemic venous muscular momentum
        self.f_[20] = q_vencer_sys_ - q_arcer_sys_                                                                             # systemic venous cerebral flow balance
        self.f_[21] = (p_ven_sys_ - p_vencer_sys_)/self.R_vencer_sys + q_vencer_sys_                                           # systemic venous cerebral momentum
        self.f_[22] = q_vencor_sys_ - q_arcor_sys_                                                                             # systemic venous coronary flow balance
        self.f_[23] = (p_ven_sys_ - p_vencor_sys_)/self.R_vencor_sys + q_vencor_sys_                                           # systemic venous coronary momentum
        self.f_[24] = sum(q_ven_sys_) - (q_venspl_sys_ + q_venespl_sys_ + q_venmsc_sys_ + q_vencer_sys_ + q_vencor_sys_)       # systemic venous flow balance
        for n in range(self.vs):
            self.f_[25+n] = (p_at_r_i_[n]-p_ven_sys_)/R_ven_sys[n] + q_ven_sys_[n]                                             # systemic venous momentum
                # -----------------------------------------------------------
        self.f_[25+self.vs] = -sum(q_ven_sys_) + q_vin_r_ - (1-self.switch_V[3]) * VQ_at_r_                                    # right atrium flow balance
        self.f_[26+self.vs] = vl_tv_ + q_vin_r_                                                                                # tricuspid valve momentum
        self.f_[27+self.vs] = -q_vin_r_ + q_vout_r_ - (1-self.switch_V[1]) * VQ_v_r_                                           # right ventricle flow balance
        self.f_[28+self.vs] = vl_pv_ + q_vout_r_                                                                               # pulmonary valve momentum
        self.f_[29+self.vs] = -q_vout_r_ + q_ar_pul_                                                                           # pulmonary arterial flow balance
        self.f_[30+self.vs] = (p_cap_pul_ - p_ar_pul_)/self.R_ar_pul + q_ar_pul_                                               # pulmonary arterial momentum
        self.f_[31+self.vs] = -q_ar_pul_ + q_cap_pul_                                                                          # pulmonary capillary flow balance
        self.f_[32+self.vs] = (p_ven_pul_ - p_cap_pul_)/self.R_cap_pul + q_cap_pul_                                            # pulmonary capillary momentum
        self.f_[33+self.vs] = -q_cap_pul_ + sum(q_ven_pul_)                                                                    # pulmonary venous flow balance
        for n in range(self.vp):
            self.f_[34+self.vs+n] = (p_at_l_i_[n] - p_ven_pul_)/R_ven_pul[n] + q_ven_pul_[n]                                   # pulmonary venous momentum


        # setup auxiliary variable map
        # coupling variables, 0D chamber volumes, compartment volumes, other shady quantities...
        nc = len(self.c_)
        for i in range(nc): self.auxmap[self.cname[i]] = i
        if self.chmodels['lv']['type']=='0D_elast' or self.chmodels['lv']['type']=='prescribed' or self.chmodels['lv']['type']=='0D_elast_prescr': self.auxmap['V_v_l'] = nc+0
        if self.chmodels['rv']['type']=='0D_elast' or self.chmodels['rv']['type']=='prescribed' or self.chmodels['rv']['type']=='0D_elast_prescr': self.auxmap['V_v_r'] = nc+1
        if self.chmodels['la']['type']=='0D_elast' or self.chmodels['la']['type']=='prescribed' or self.chmodels['la']['type']=='0D_elast_prescr': self.auxmap['V_at_l'] = nc+2
        if self.chmodels['ra']['type']=='0D_elast' or self.chmodels['ra']['type']=='prescribed' or self.chmodels['ra']['type']=='0D_elast_prescr': self.auxmap['V_at_r'] = nc+3
        self.auxmap['V_ar_sys'] = nc+4
        self.auxmap['V_arperi_sys'] = nc+5
        self.auxmap['V_venspl_sys'] = nc+6
        self.auxmap['V_venespl_sys'] = nc+7
        self.auxmap['V_venmsc_sys'] = nc+8
        self.auxmap['V_vencer_sys'] = nc+9
        self.auxmap['V_vencor_sys'] = nc+10
        self.auxmap['V_ven_sys'] = nc+11
        self.auxmap['V_ar_pul'] = nc+12
        self.auxmap['V_cap_pul'] = nc+13
        self.auxmap['V_ven_pul'] = nc+14

        # populate auxiliary variable vector
        for i in range(nc): self.a_[i] = self.c_[i]
        self.a_[nc+0]  = VQ_v_l_ * self.switch_V[0]
        self.a_[nc+1]  = VQ_v_r_ * self.switch_V[1]
        self.a_[nc+2]  = VQ_at_l_ * self.switch_V[2]
        self.a_[nc+3]  = VQ_at_r_ * self.switch_V[3]
        self.a_[nc+4]  = self.C_ar_sys * p_ard_sys_ + self.V_ar_sys_u
        self.a_[nc+5]  = (self.C_arspl_sys+self.C_arespl_sys+self.C_armsc_sys+self.C_arcer_sys+self.C_arcor_sys) * p_arperi_sys_ + self.V_arspl_sys_u+self.V_arespl_sys_u+self.V_armsc_sys_u+self.V_arcer_sys_u+self.V_arcor_sys_u
        self.a_[nc+6]  = self.C_venspl_sys * p_venspl_sys_ + self.V_venspl_sys_u
        self.a_[nc+7]  = self.C_venespl_sys * p_venespl_sys_ + self.V_venespl_sys_u
        self.a_[nc+8]  = self.C_venmsc_sys * p_venmsc_sys_ + self.V_venmsc_sys_u
        self.a_[nc+9]  = self.C_vencer_sys * p_vencer_sys_ + self.V_vencer_sys_u
        self.a_[nc+10] = self.C_vencor_sys * p_vencor_sys_ + self.V_vencor_sys_u
        self.a_[nc+11] = self.C_ven_sys * p_ven_sys_ + self.V_ven_sys_u
        self.a_[nc+12] = self.C_ar_pul * p_ar_pul_ + self.V_ar_pul_u
        self.a_[nc+13] = self.C_cap_pul * p_cap_pul_ + self.V_cap_pul_u
        self.a_[nc+14] = self.C_ven_pul * p_ven_pul_ + self.V_ven_pul_u



    def initialize(self, var, iniparam):

        var[self.varmap['q_vin_l']]                           = iniparam['q_vin_l_0']
        var[self.varmap[self.vname[2]]]                       = iniparam[self.vname[2]+'_0']
        var[self.varmap['q_vout_l']]                          = iniparam['q_vout_l_0']
        var[self.varmap[self.vname[0]]]                       = iniparam[self.vname[0]+'_0']
        var[self.varmap['p_ar_sys']]                          = iniparam['p_ar_sys_0']
        try: var[self.varmap['q_arp_sys']]                    = iniparam['q_arp_sys_0']
        except: var[self.varmap['q_arp_sys']]                 = iniparam['q_ar_sys_0']
        try: var[self.varmap['p_ard_sys']]                    = iniparam['p_ard_sys_0']
        except: var[self.varmap['p_ard_sys']]                 = iniparam['p_ar_sys_0']
        var[self.varmap['q_ar_sys']]                          = iniparam['q_ar_sys_0']
        var[self.varmap['p_arperi_sys']]                      = iniparam['p_arperi_sys_0']
        var[self.varmap['q_arspl_sys']]                       = iniparam['q_arspl_sys_0']
        var[self.varmap['q_arespl_sys']]                      = iniparam['q_arespl_sys_0']
        var[self.varmap['q_armsc_sys']]                       = iniparam['q_armsc_sys_0']
        var[self.varmap['q_arcer_sys']]                       = iniparam['q_arcer_sys_0']
        var[self.varmap['q_arcor_sys']]                       = iniparam['q_arcor_sys_0']
        var[self.varmap['p_venspl_sys']]                      = iniparam['p_venspl_sys_0']
        var[self.varmap['q_venspl_sys']]                      = iniparam['q_venspl_sys_0']
        var[self.varmap['p_venespl_sys']]                     = iniparam['p_venespl_sys_0']
        var[self.varmap['q_venespl_sys']]                     = iniparam['q_venespl_sys_0']
        var[self.varmap['p_venmsc_sys']]                      = iniparam['p_venmsc_sys_0']
        var[self.varmap['q_venmsc_sys']]                      = iniparam['q_venmsc_sys_0']
        var[self.varmap['p_vencer_sys']]                      = iniparam['p_vencer_sys_0']
        var[self.varmap['q_vencer_sys']]                      = iniparam['q_vencer_sys_0']
        var[self.varmap['p_vencor_sys']]                      = iniparam['p_vencor_sys_0']
        var[self.varmap['q_vencor_sys']]                      = iniparam['q_vencor_sys_0']
        var[self.varmap['p_ven_sys']]                         = iniparam['p_ven_sys_0']
        for n in range(self.vs):
            try: var[self.varmap['q_ven'+str(n+1)+'_sys']]    = iniparam['q_ven'+str(n+1)+'_sys_0']
            except: var[self.varmap['q_ven'+str(n+1)+'_sys']]    = iniparam['q_ven_sys_0']
        var[self.varmap['q_vin_r']]                           = iniparam['q_vin_r_0']
        var[self.varmap[self.vname[3]]]                       = iniparam[self.vname[3]+'_0']
        var[self.varmap['q_vout_r']]                          = iniparam['q_vout_r_0']
        var[self.varmap[self.vname[1]]]                       = iniparam[self.vname[1]+'_0']
        var[self.varmap['p_ar_pul']]                          = iniparam['p_ar_pul_0']
        var[self.varmap['q_ar_pul']]                          = iniparam['q_ar_pul_0']
        var[self.varmap['p_cap_pul']]                         = iniparam['p_cap_pul_0']
        var[self.varmap['q_cap_pul']]                         = iniparam['q_cap_pul_0']
        var[self.varmap['p_ven_pul']]                         = iniparam['p_ven_pul_0']
        for n in range(self.vp):
            try: var[self.varmap['q_ven'+str(n+1)+'_pul']]    = iniparam['q_ven'+str(n+1)+'_pul_0']
            except: var[self.varmap['q_ven'+str(n+1)+'_pul']]    = iniparam['q_ven_pul_0']


    def check_periodic(self, varTc, varTc_old, auxTc, auxTc_old, eps, check, cyclerr):

        if isinstance(varTc, np.ndarray): varTc_sq, varTc_old_sq = varTc, varTc_old
        else: varTc_sq, varTc_old_sq = allgather_vec(varTc, self.comm), allgather_vec(varTc_old, self.comm)

        vals = []

        if check[0]=='allvar':

            var_ids, aux_ids = list(self.varmap.values()), []

        elif check[0]=='allvaraux':

            var_ids, aux_ids = list(self.varmap.values()), list(self.auxmap.values())

        elif check[0]=='pQvar':

            var_ids, aux_ids = [self.varmap[self.vname[2]],self.varmap[self.vname[0]],self.varmap['p_ar_sys'],self.varmap['p_ard_sys'],self.varmap['p_arperi_sys'],self.varmap['p_venspl_sys'],self.varmap['p_venespl_sys'],self.varmap['p_venmsc_sys'],self.varmap['p_vencer_sys'],self.varmap['p_vencor_sys'],self.varmap['p_ven_sys'],self.varmap[self.vname[3]],self.varmap[self.vname[1]],self.varmap['p_ar_pul'],self.varmap['p_cap_pul'],self.varmap['p_ven_pul']], []

        elif check[0]=='specific':

            var_ids = []
            for k in range(len(self.varmap)):
                if list(self.varmap.keys())[k] in check[1]:
                    var_ids.append(list(self.varmap.values())[k])

            aux_ids = []
            for k in range(len(self.auxmap)):
                if list(self.auxmap.keys())[k] in check[1]:
                    aux_ids.append(list(self.auxmap.values())[k])

        else:
            raise NameError("Unknown check option!")

        # compute the errors
        for i in range(len(varTc_sq)):
            if i in var_ids:
                vals.append( math.fabs((varTc_sq[i]-varTc_old_sq[i])/max(1.0,math.fabs(varTc_old_sq[i]))) )

        for i in range(len(auxTc)):
            if i in aux_ids:
                vals.append( math.fabs((auxTc[i]-auxTc_old[i])/max(1.0,math.fabs(auxTc_old[i]))) )

        cyclerr[0] = max(vals)

        if cyclerr[0] <= eps:
            is_periodic = True
        else:
            is_periodic = False

        if not isinstance(varTc, np.ndarray): del varTc_sq, varTc_old_sq

        return is_periodic


    def print_to_screen(self, var, aux):

        if isinstance(var, np.ndarray): var_sq = var
        else: var_sq = allgather_vec(var, self.comm)

        nc = len(self.c_)

        if self.comm.rank == 0:

            print("Output of 0D vascular model (syspulcap):")

            for i in range(nc):
                print('{:<12s}{:<3s}{:<10.3f}'.format(self.cname[i],' = ',aux[self.auxmap[self.cname[i]]]))

            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format(self.vname[2],' = ',var_sq[self.varmap[self.vname[2]]],'   ',self.vname[3],' = ',var_sq[self.varmap[self.vname[3]]]))
            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format(self.vname[0],' = ',var_sq[self.varmap[self.vname[0]]],'   ',self.vname[1],' = ',var_sq[self.varmap[self.vname[1]]]))

            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format('p_ar_sys',' = ',var_sq[self.varmap['p_ar_sys']],'   ','p_ar_pul',' = ',var_sq[self.varmap['p_ar_pul']]))
            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format('p_arperi_sys',' = ',var_sq[self.varmap['p_arperi_sys']],'   ','p_cap_pul',' = ',var_sq[self.varmap['p_cap_pul']]))
            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format('p_ven_sys',' = ',var_sq[self.varmap['p_ven_sys']],'   ','p_ven_pul',' = ',var_sq[self.varmap['p_ven_pul']]))

            sys.stdout.flush()

        if not isinstance(var, np.ndarray): del var_sq




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
    groups.append({'vol_time_compart' : ['V_at_l', 'V_v_l', 'V_at_r', 'V_v_r', 'V_ar_sys', 'V_arperi_sys', 'V_venspl_sys', 'V_venespl_sys', 'V_venmsc_sys', 'V_vencer_sys', 'V_vencor_sys', 'V_ven_sys', 'V_ar_pul', 'V_cap_pul', 'V_ven_pul'],
                'tex'                 : ['$V_{\\\mathrm{at}}^{\\\ell}$', '$V_{\\\mathrm{v}}^{\\\ell}$', '$V_{\\\mathrm{at}}^{r}$', '$V_{\\\mathrm{v}}^{r}$', '$V_{\\\mathrm{ar}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ar,peri}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven,spl}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven,espl}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven,msc}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven,cer}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven,cor}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ar}}^{\\\mathrm{pul}}$', '$V_{\\\mathrm{cap}}^{\\\mathrm{pul}}$', '$V_{\\\mathrm{ven}}^{\\\mathrm{pul}}$'],
                'lines'               : [1, 2, 16, 17, 3, 4, 10, 11, 12, 13, 14, 15, 18, 19, 20]})

    # all volumes summed up for conservation check
    groups[5]['vol_time_compart'].append('V_all')
    groups[5]['tex'].append('$\\\sum V$')
    groups[5]['lines'].append(99)

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







# similar to syspulcap model, however with the coronaries branching off after the aortic valve and directly feeding back into the right atrium

# 37 governing equations (uncomment and paste directly into a LaTeX environment):

#% left heart and systemic circulation:
#\begin{align}
#&-Q_{\mathrm{at}}^{\ell} = q_{\mathrm{ven}}^{\mathrm{pul}} - q_{\mathrm{v,in}}^{\ell}\nonumber\\
#&\tilde{R}_{\mathrm{v,in}}^{\ell}\,q_{\mathrm{v,in}}^{\ell} = p_{\mathrm{at}}^{\ell}-p_{\mathrm{v}}^{\ell}\nonumber\\
#&-Q_{\mathrm{v}}^{\ell} = q_{\mathrm{v,in}}^{\ell} - q_{\mathrm{v,out}}^{\ell}\nonumber\\
#&\tilde{R}_{\mathrm{v,out}}^{\ell}\,q_{\mathrm{v,out}}^{\ell} = p_{\mathrm{v}}^{\ell}-p_{\mathrm{ar}}^{\mathrm{sys}}\nonumber\\
#&0 = q_{\mathrm{v,out}}^{\ell} - q_{\mathrm{ar,p}}^{\mathrm{sys}} - q_{\mathrm{ar,cor,in}}^{\mathrm{sys}}\nonumber\\
#&I_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}q_{\mathrm{ar,p}}^{\mathrm{sys}}}{\mathrm{d}t} + Z_{\mathrm{ar}}^{\mathrm{sys}}\,q_{\mathrm{ar,p}}^{\mathrm{sys}}=p_{\mathrm{ar}}^{\mathrm{sys}}-p_{\mathrm{ar,d}}^{\mathrm{sys}}\nonumber\\
#&C_{\mathrm{ar,cor}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar,cor,in}}^{\mathrm{sys}} - q_{\mathrm{ar,cor}}^{\mathrm{sys}}\nonumber\\
#&R_{\mathrm{ar,cor}}^{\mathrm{sys}}\,q_{\mathrm{ar,cor}}^{\mathrm{sys}} = p_{\mathrm{ar}}^{\mathrm{sys}} - p_{\mathrm{ven,cor}}^{\mathrm{sys}}\nonumber\\
#&C_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ar,d}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar,p}}^{\mathrm{sys}} - q_{\mathrm{ar}}^{\mathrm{sys}}\nonumber\\
#&L_{\mathrm{ar}}^{\mathrm{sys}}\frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{sys}}\,q_{\mathrm{ar}}^{\mathrm{sys}}=p_{\mathrm{ar,d}}^{\mathrm{sys}} -p_{\mathrm{ar,peri}}^{\mathrm{sys}}\nonumber\\
#&\left(\sum_{j\in\{\mathrm{spl,espl,\atop msc,cer}\}}\!\!\!\!\!\!\!\!\!C_{\mathrm{ar},j}^{\mathrm{sys}}\right) \frac{\mathrm{d}p_{\mathrm{ar,peri}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{sys}}-\!\!\!\!\!\sum_{j\in\{\mathrm{spl,espl,\atop msc,cer}\}}\!\!\!\!\!\!\!\!\!q_{\mathrm{ar},j}^{\mathrm{sys}}\nonumber\\
#&R_{\mathrm{ar},i}^{\mathrm{sys}}\,q_{\mathrm{ar},i}^{\mathrm{sys}} = p_{\mathrm{ar,peri}}^{\mathrm{sys}} - p_{\mathrm{ven},i}^{\mathrm{sys}}, \quad\scriptstyle{i\in\{\mathrm{spl,espl,\atop msc,cer}\}}\nonumber\\
#&C_{\mathrm{ven},i}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven},i}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar},i}^{\mathrm{sys}} - q_{\mathrm{ven},i}^{\mathrm{sys}}, \quad\scriptstyle{i\in\{\mathrm{spl,espl,\atop msc,cer}\}}\nonumber\\
#&R_{\mathrm{ven},i}^{\mathrm{sys}}\,q_{\mathrm{ven},i}^{\mathrm{sys}} = p_{\mathrm{ven},i}^{\mathrm{sys}}-p_{\mathrm{ven}}^{\mathrm{sys}}, \quad\scriptstyle{i\in\{\mathrm{spl,espl,\atop msc,cer}\}}\nonumber\\
#&C_{\mathrm{ven}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{sys}}}{\mathrm{d}t} = \!\!\!\!\sum_{j=\mathrm{spl,espl,\atop msc,cer}}\!\!\!\!\!q_{\mathrm{ven},j}^{\mathrm{sys}}-q_{\mathrm{ven}}^{\mathrm{sys}}\nonumber\\
#&L_{\mathrm{ven}}^{\mathrm{sys}}\frac{\mathrm{d}q_{\mathrm{ven}}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ven}}^{\mathrm{sys}}\, q_{\mathrm{ven}}^{\mathrm{sys}} = p_{\mathrm{ven}}^{\mathrm{sys}} - p_{\mathrm{at}}^{r}\nonumber\\
#&C_{\mathrm{ven,cor}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven,cor}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar,cor}}^{\mathrm{sys}}-q_{\mathrm{ven,cor}}^{\mathrm{sys}}\nonumber\\
#&R_{\mathrm{ven,cor}}^{\mathrm{sys}}\,q_{\mathrm{ven,cor}}^{\mathrm{sys}} = p_{\mathrm{ven,cor}}^{\mathrm{sys}} - p_{\mathrm{at}}^{r}\nonumber
#\end{align}

#% right heart and pulmonary circulation:
#\begin{align}
#&-Q_{\mathrm{at}}^{r} = q_{\mathrm{ven}}^{\mathrm{sys}} + q_{\mathrm{ven,cor}}^{\mathrm{sys}} - q_{\mathrm{v,in}}^{r}\nonumber\\
#&\tilde{R}_{\mathrm{v,in}}^{r}\,q_{\mathrm{v,in}}^{r} = p_{\mathrm{at}}^{r}-p_{\mathrm{v}}^{r}\nonumber\\
#&-Q_{\mathrm{v}}^{r} = q_{\mathrm{v,in}}^{r} - q_{\mathrm{v,out}}^{r}\nonumber\\
#&\tilde{R}_{\mathrm{v,out}}^{r}\,q_{\mathrm{v,out}}^{r} = p_{\mathrm{v}}^{r}-p_{\mathrm{ar}}^{\mathrm{pul}}\nonumber\\
#&C_{\mathrm{ar}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{v,out}}^{r} - q_{\mathrm{ar}}^{\mathrm{pul}}\nonumber\\
#&L_{\mathrm{ar}}^{\mathrm{pul}}\frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{pul}}\,q_{\mathrm{ar}}^{\mathrm{pul}}=p_{\mathrm{ar}}^{\mathrm{pul}} -p_{\mathrm{cap}}^{\mathrm{pul}}\nonumber\\
#&C_{\mathrm{cap}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{cap}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{pul}} - q_{\mathrm{cap}}^{\mathrm{pul}}\nonumber\\
#&R_{\mathrm{cap}}^{\mathrm{pul}}\,q_{\mathrm{cap}}^{\mathrm{pul}}=p_{\mathrm{cap}}^{\mathrm{pul}}-p_{\mathrm{ven}}^{\mathrm{pul}}\nonumber\\
#&C_{\mathrm{ven}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{cap}}^{\mathrm{pul}} - q_{\mathrm{ven}}^{\mathrm{pul}}\nonumber\\
#&L_{\mathrm{ven}}^{\mathrm{pul}}\frac{\mathrm{d}q_{\mathrm{ven}}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ven}}^{\mathrm{pul}}\, q_{\mathrm{ven}}^{\mathrm{pul}}=p_{\mathrm{ven}}^{\mathrm{pul}}-p_{\mathrm{at}}^{\ell}\nonumber
#\end{align}

class cardiovascular0Dsyspulcapcor(cardiovascular0Dsyspulcap):

    def setup_arrays(self):

        # number of degrees of freedom
        self.numdof = 31 + self.vs + self.vp

        self.elastarrays = [[]]*4

        self.si, self.switch_V = [0]*5, [1]*5 # default values

        self.vindex_ch = [3,25+self.vs,1,23+self.vs] # coupling variable indices (decreased by 1 for pressure coupling!)
        self.vname, self.cname = ['p_v_l','p_v_r','p_at_l','p_at_r'], []

        # set those ids which are relevant for monolithic direct coupling
        self.v_ids, self.c_ids = [], []
        self.cindex_ch = [2,24+self.vs,0,22+self.vs]

        self.switch_cor = 0

        if self.cormodel is not None:

            # additional venous inflow is added by coronary model, so make sure that vs is one less if user requests a 3D RA model
            if 'num_inflows' in self.chmodels['ra'].keys() and self.vs > 1: self.vs -= 1

            # initialize coronary circulation model
            if self.cormodel == 'ZCRp_CRd':
                from .cardiovascular0D_coronary import coronary_circ_ZCRp_CRd
                self.corcirc = coronary_circ_ZCRp_CRd(self.params, self.varmap, self.auxmap, self.vs)
            elif self.cormodel == 'ZCRp_CRd_lr':
                from .cardiovascular0D_coronary import coronary_circ_ZCRp_CRd_lr
                self.corcirc = coronary_circ_ZCRp_CRd_lr(self.params, self.varmap, self.auxmap, self.vs)
            else:
                raise NameError("Unknown coronary circulation model!")

            self.switch_cor = 1

            self.numdof += self.corcirc.ndcor

        self.set_solve_arrays()


    def equation_map(self):

        # variable map
        self.varmap['q_vin_l']                   = 0+self.si[2]
        self.varmap[self.vname[2]]               = 1-self.si[2]
        self.varmap['q_vout_l']                  = 2+self.si[0]
        self.varmap[self.vname[0]]               = 3-self.si[0]
        self.varmap['p_ar_sys']                  = 4
        self.varmap['q_arp_sys']                 = 5
        self.varmap['p_ard_sys']                 = 6
        self.varmap['q_ar_sys']                  = 7
        self.varmap['p_arperi_sys']              = 8
        self.varmap['q_arspl_sys']               = 9
        self.varmap['q_arespl_sys']              = 10
        self.varmap['q_armsc_sys']               = 11
        self.varmap['q_arcer_sys']               = 12
        self.varmap['p_venspl_sys']              = 13
        self.varmap['q_venspl_sys']              = 14
        self.varmap['p_venespl_sys']             = 15
        self.varmap['q_venespl_sys']             = 16
        self.varmap['p_venmsc_sys']              = 17
        self.varmap['q_venmsc_sys']              = 18
        self.varmap['p_vencer_sys']              = 19
        self.varmap['q_vencer_sys']              = 20
        self.varmap['p_ven_sys']                 = 21
        for n in range(self.vs):
            self.varmap['q_ven'+str(n+1)+'_sys'] = 22+n
        self.varmap['q_vin_r']                   = 22+self.vs+self.si[3]
        self.varmap[self.vname[3]]               = 23+self.vs-self.si[3]
        self.varmap['q_vout_r']                  = 24+self.vs+self.si[1]
        self.varmap[self.vname[1]]               = 25+self.vs-self.si[1]
        self.varmap['p_ar_pul']                  = 26+self.vs
        self.varmap['q_ar_pul']                  = 27+self.vs
        self.varmap['p_cap_pul']                 = 28+self.vs
        self.varmap['q_cap_pul']                 = 29+self.vs
        self.varmap['p_ven_pul']                 = 30+self.vs
        for n in range(self.vp):
            self.varmap['q_ven'+str(n+1)+'_pul'] = 31+self.vs+n

        q_ven_sys_, q_ven_pul_ = [], []
        p_at_l_i_, p_at_r_i_ = [], []

        self.t_            = sp.Symbol('t_')
        q_vin_l_           = sp.Symbol('q_vin_l_')
        for n in range(self.vp): p_at_l_i_.append(sp.Symbol('p_at_l_i'+str(n+1)+'_'))
        p_at_l_o1_         = sp.Symbol('p_at_l_o1_')
        q_vout_l_          = sp.Symbol('q_vout_l_')
        p_v_l_i1_, p_v_l_o1_ = sp.Symbol('p_v_l_i1_'), sp.Symbol('p_v_l_o1_')
        p_ar_sys_          = sp.Symbol('p_ar_sys_')
        q_arp_sys_      = sp.Symbol('q_arp_sys_')
        p_ard_sys_      = sp.Symbol('p_ard_sys_')
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
        p_ven_sys_         = sp.Symbol('p_ven_sys_')
        for n in range(self.vs): q_ven_sys_.append(sp.Symbol('q_ven'+str(n+1)+'_sys_'))
        q_vin_r_           = sp.Symbol('q_vin_r_')
        for n in range(self.vs+self.switch_cor): p_at_r_i_.append(sp.Symbol('p_at_r_i'+str(n+1)+'_'))
        p_at_r_o1_         = sp.Symbol('p_at_r_o1_')
        q_vout_r_          = sp.Symbol('q_vout_r_')
        p_v_r_i1_, p_v_r_o1_ = sp.Symbol('p_v_r_i1_'), sp.Symbol('p_v_r_o1_')
        p_ar_pul_          = sp.Symbol('p_ar_pul_')
        q_ar_pul_          = sp.Symbol('q_ar_pul_')
        p_cap_pul_         = sp.Symbol('p_cap_pul_')
        q_cap_pul_         = sp.Symbol('q_cap_pul_')
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
        self.x_[self.varmap['q_vin_l']]                   = q_vin_l_
        self.x_[self.varmap[self.vname[2]]]               = p_at_l_i_[0]
        self.x_[self.varmap['q_vout_l']]                  = q_vout_l_
        self.x_[self.varmap[self.vname[0]]]               = p_v_l_i1_
        self.x_[self.varmap['p_ar_sys']]                  = p_ar_sys_
        self.x_[self.varmap['q_arp_sys']]                 = q_arp_sys_
        self.x_[self.varmap['p_ard_sys']]                 = p_ard_sys_
        self.x_[self.varmap['q_ar_sys']]                  = q_ar_sys_
        self.x_[self.varmap['p_arperi_sys']]              = p_arperi_sys_
        self.x_[self.varmap['q_arspl_sys']]               = q_arspl_sys_
        self.x_[self.varmap['q_arespl_sys']]              = q_arespl_sys_
        self.x_[self.varmap['q_armsc_sys']]               = q_armsc_sys_
        self.x_[self.varmap['q_arcer_sys']]               = q_arcer_sys_
        self.x_[self.varmap['p_venspl_sys']]              = p_venspl_sys_
        self.x_[self.varmap['q_venspl_sys']]              = q_venspl_sys_
        self.x_[self.varmap['p_venespl_sys']]             = p_venespl_sys_
        self.x_[self.varmap['q_venespl_sys']]             = q_venespl_sys_
        self.x_[self.varmap['p_venmsc_sys']]              = p_venmsc_sys_
        self.x_[self.varmap['q_venmsc_sys']]              = q_venmsc_sys_
        self.x_[self.varmap['p_vencer_sys']]              = p_vencer_sys_
        self.x_[self.varmap['q_vencer_sys']]              = q_vencer_sys_
        self.x_[self.varmap['p_ven_sys']]                 = p_ven_sys_
        for n in range(self.vs):
            self.x_[self.varmap['q_ven'+str(n+1)+'_sys']] = q_ven_sys_[n]
        self.x_[self.varmap['q_vin_r']]                   = q_vin_r_
        self.x_[self.varmap[self.vname[3]]]               = p_at_r_i_[0]
        self.x_[self.varmap['q_vout_r']]                  = q_vout_r_
        self.x_[self.varmap[self.vname[1]]]               = p_v_r_i1_
        self.x_[self.varmap['p_ar_pul']]                  = p_ar_pul_
        self.x_[self.varmap['q_ar_pul']]                  = q_ar_pul_
        self.x_[self.varmap['p_cap_pul']]                 = p_cap_pul_
        self.x_[self.varmap['q_cap_pul']]                 = q_cap_pul_
        self.x_[self.varmap['p_ven_pul']]                 = p_ven_pul_
        for n in range(self.vp):
            self.x_[self.varmap['q_ven'+str(n+1)+'_pul']] = q_ven_pul_[n]


        # set chamber dicts
        chdict_lv = {'VQ' : VQ_v_l_, 'pi1' : p_v_l_i1_, 'po1' : p_v_l_o1_}
        chdict_rv = {'VQ' : VQ_v_r_, 'pi1' : p_v_r_i1_, 'po1' : p_v_r_o1_}
        chdict_la = {'VQ' : VQ_at_l_, 'po1' : p_at_l_o1_}
        for n in range(self.vp): chdict_la['pi'+str(n+1)] = p_at_l_i_[n]
        chdict_ra = {'VQ' : VQ_at_r_, 'po1' : p_at_r_o1_}
        for n in range(self.vs+self.switch_cor): chdict_ra['pi'+str(n+1)] = p_at_r_i_[n]

        # set coupling states and variables (e.g., express V in terms of p and E in case of elastance models, ...)
        self.set_coupling_state('lv', chdict_lv, [E_v_l_])
        self.set_coupling_state('rv', chdict_rv, [E_v_r_])
        self.set_coupling_state('la', chdict_la, [E_at_l_])
        self.set_coupling_state('ra', chdict_ra, [E_at_r_])

        # feed back modified dicts to chamber variables
        VQ_v_l_, p_v_l_i1_, p_v_l_o1_ = chdict_lv['VQ'], chdict_lv['pi1'], chdict_lv['po1']
        VQ_v_r_, p_v_r_i1_, p_v_r_o1_ = chdict_rv['VQ'], chdict_rv['pi1'], chdict_rv['po1']
        VQ_at_l_, p_ati1_l_, p_at_l_o1_ = chdict_la['VQ'], chdict_la['pi1'], chdict_la['po1']
        for n in range(self.vp): p_at_l_i_[n] = chdict_la['pi'+str(n+1)]
        VQ_at_r_, p_ati1_r_, p_at_r_o1_ = chdict_ra['VQ'], chdict_ra['pi1'], chdict_ra['po1']
        for n in range(self.vs+self.switch_cor): p_at_r_i_[n] = chdict_ra['pi'+str(n+1)]

        # add coronary circulation equations
        if self.cormodel is not None: # p_v_l on purpose here NOT passed to coronary model (for comparison reasons to syspulcap model...)
            q_arcor_sys_in_, q_vencor_sys_out_ = self.corcirc.equation_map(self.numdof-self.corcirc.ndcor, len(self.c_)+14, self.x_, self.a_, self.df_, self.f_, [p_ar_sys_], 0, p_at_r_i_[-1])
        else:
            q_arcor_sys_in_, q_vencor_sys_out_ = [0], 0

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
        self.df_[0]  = VQ_at_l_ * self.switch_V[2]                                                            # left atrium volume rate
        self.df_[1]  = (self.L_vin_l/R_vin_l_) * q_vin_l_                                                     # mitral valve inertia
        self.df_[2]  = VQ_v_l_ * self.switch_V[0]                                                             # left ventricle volume rate
        self.df_[3]  = (self.L_vout_l/R_vout_l_) * q_vout_l_                                                  # aortic valve inertia
        self.df_[4]  = 0.
        self.df_[5]  = (self.I_ar_sys/self.Z_ar_sys) * q_arp_sys_                                          # aortic root inertia
        self.df_[6]  = self.C_ar_sys * p_ard_sys_                                                          # systemic arterial volume rate
        self.df_[7]  = (self.L_ar_sys/self.R_ar_sys) * q_ar_sys_                                              # systemic arterial inertia
        self.df_[8]  = (self.C_arspl_sys+self.C_arespl_sys+self.C_armsc_sys+self.C_arcer_sys) * p_arperi_sys_ # systemic capillary volume rate
        self.df_[9]  = 0.
        self.df_[10] = 0.
        self.df_[11] = 0.
        self.df_[12] = 0.
        self.df_[13] = self.C_venspl_sys * p_venspl_sys_                                                      # systemic venous splanchnic volume rate
        self.df_[14] = 0.
        self.df_[15] = self.C_venespl_sys * p_venespl_sys_                                                    # systemic venous extra-splanchnic volume rate
        self.df_[16] = 0.
        self.df_[17] = self.C_venmsc_sys * p_venmsc_sys_                                                      # systemic venous muscular volume rate
        self.df_[18] = 0.
        self.df_[19] = self.C_vencer_sys * p_vencer_sys_                                                      # systemic venous cerebral volume rate
        self.df_[20] = 0.
        self.df_[21] = self.C_ven_sys * p_ven_sys_                                                            # systemic venous volume rate
        for n in range(self.vs):
            self.df_[22+n] = (L_ven_sys[n]/R_ven_sys[n]) * q_ven_sys_[n]                                      # systemic venous inertia
                # -----------------------------------------------------------
        self.df_[22+self.vs] = VQ_at_r_ * self.switch_V[3]                                                    # right atrium volume rate
        self.df_[23+self.vs] = (self.L_vin_r/R_vin_r_) * q_vin_r_                                             # tricuspid valve inertia
        self.df_[24+self.vs] = VQ_v_r_ * self.switch_V[1]                                                     # right ventricle volume rate
        self.df_[25+self.vs] = (self.L_vout_r/R_vout_r_) * q_vout_r_                                          # pulmonary valve inertia
        self.df_[26+self.vs] = self.C_ar_pul * p_ar_pul_                                                      # pulmonary arterial volume rate
        self.df_[27+self.vs] = (self.L_ar_pul/self.R_ar_pul) * q_ar_pul_                                      # pulmonary arterial inertia
        self.df_[28+self.vs] = self.C_cap_pul * p_cap_pul_                                                    # pulmonary capillary volume rate
        self.df_[29+self.vs] = 0.
        self.df_[30+self.vs] = self.C_ven_pul * p_ven_pul_                                                    # pulmonary venous volume rate
        for n in range(self.vp):
            self.df_[31+self.vs+n] = (L_ven_pul[n]/R_ven_pul[n]) * q_ven_pul_[n]                              # pulmonary venous inertia


        # f part of rhs contribution theta * f + (1-theta) * f_old
        self.f_[0]  = -sum(q_ven_pul_) + q_vin_l_ - (1-self.switch_V[2]) * VQ_at_l_                           # left atrium flow balance
        self.f_[1]  = vl_mv_ + q_vin_l_                                                                       # mitral valve momentum
        self.f_[2]  = -q_vin_l_ + q_vout_l_ - (1-self.switch_V[0]) * VQ_v_l_                                  # left ventricle flow balance
        self.f_[3]  = vl_av_ + q_vout_l_                                                                      # aortic valve momentum
        self.f_[4]  = -q_vout_l_ + q_arp_sys_ + self.switch_cor * sum(q_arcor_sys_in_)                     # aortic root flow balance
        self.f_[5]  = (p_ard_sys_ - p_ar_sys_)/self.Z_ar_sys + q_arp_sys_                               # aortic root momentum
        self.f_[6]  = -q_arp_sys_ + q_ar_sys_                                                              # systemic arterial flow balance
        self.f_[7]  = (p_arperi_sys_ - p_ard_sys_)/self.R_ar_sys + q_ar_sys_                               # systemic arterial momentum
        self.f_[8]  = -q_ar_sys_ + (q_arspl_sys_ + q_arespl_sys_ + q_armsc_sys_ + q_arcer_sys_)               # systemic capillary flow balance
        self.f_[9]  = (p_venspl_sys_ - p_arperi_sys_)/self.R_arspl_sys + q_arspl_sys_                         # systemic arterial splanchnic momentum
        self.f_[10] = (p_venespl_sys_ - p_arperi_sys_)/self.R_arespl_sys + q_arespl_sys_                      # systemic arterial extra-splanchnic momentum
        self.f_[11] = (p_venmsc_sys_ - p_arperi_sys_)/self.R_armsc_sys + q_armsc_sys_                         # systemic arterial muscular momentum
        self.f_[12] = (p_vencer_sys_ - p_arperi_sys_)/self.R_arcer_sys + q_arcer_sys_                         # systemic arterial cerebral momentum
        self.f_[13] = q_venspl_sys_ - q_arspl_sys_                                                            # systemic venous splanchnic flow balance
        self.f_[14] = (p_ven_sys_ - p_venspl_sys_)/self.R_venspl_sys + q_venspl_sys_                          # systemic venous splanchnic momentum
        self.f_[15] = q_venespl_sys_ - q_arespl_sys_                                                          # systemic venous extra-splanchnic flow balance
        self.f_[16] = (p_ven_sys_ - p_venespl_sys_)/self.R_venespl_sys + q_venespl_sys_                       # systemic venous extra-splanchnic momentum
        self.f_[17] = q_venmsc_sys_ - q_armsc_sys_                                                            # systemic venous muscular flow balance
        self.f_[18] = (p_ven_sys_ - p_venmsc_sys_)/self.R_venmsc_sys + q_venmsc_sys_                          # systemic venous muscular momentum
        self.f_[19] = q_vencer_sys_ - q_arcer_sys_                                                            # systemic venous cerebral flow balance
        self.f_[20] = (p_ven_sys_ - p_vencer_sys_)/self.R_vencer_sys + q_vencer_sys_                          # systemic venous cerebral momentum
        self.f_[21] = sum(q_ven_sys_) - (q_venspl_sys_ + q_venespl_sys_ + q_venmsc_sys_ + q_vencer_sys_)      # systemic venous flow balance
        for n in range(self.vs):
            self.f_[22+n] = (p_at_r_i_[n]-p_ven_sys_)/R_ven_sys[n] + q_ven_sys_[n]                            # systemic venous momentum
                # -----------------------------------------------------------
        self.f_[22+self.vs] = -sum(q_ven_sys_) - self.switch_cor * q_vencor_sys_out_ + q_vin_r_ - (1-self.switch_V[3]) * VQ_at_r_ # right atrium flow balance: all veins incl. coronary sinus (if present)
        self.f_[23+self.vs] = vl_tv_ + q_vin_r_                                                               # tricuspid valve momentum
        self.f_[24+self.vs] = -q_vin_r_ + q_vout_r_ - (1-self.switch_V[1]) * VQ_v_r_                          # right ventricle flow balance
        self.f_[25+self.vs] = vl_pv_ + q_vout_r_                                                              # pulmonary valve momentum
        self.f_[26+self.vs] = -q_vout_r_ + q_ar_pul_                                                          # pulmonary arterial flow balance
        self.f_[27+self.vs] = (p_cap_pul_ - p_ar_pul_)/self.R_ar_pul + q_ar_pul_                              # pulmonary arterial momentum
        self.f_[28+self.vs] = -q_ar_pul_ + q_cap_pul_                                                         # pulmonary capillary flow balance
        self.f_[29+self.vs] = (p_ven_pul_ - p_cap_pul_)/self.R_cap_pul + q_cap_pul_                           # pulmonary capillary momentum
        self.f_[30+self.vs] = -q_cap_pul_ + sum(q_ven_pul_)                                                   # pulmonary venous flow balance
        for n in range(self.vp):
            self.f_[31+self.vs+n] = (p_at_l_i_[n] - p_ven_pul_)/R_ven_pul[n] + q_ven_pul_[n]                  # pulmonary venous momentum


        # setup auxiliary variable map
        # coupling variables, 0D chamber volumes, compartment volumes, other shady quantities...
        nc = len(self.c_)
        for i in range(nc): self.auxmap[self.cname[i]] = i
        if self.chmodels['lv']['type']=='0D_elast' or self.chmodels['lv']['type']=='prescribed' or self.chmodels['lv']['type']=='0D_elast_prescr': self.auxmap['V_v_l'] = nc+0
        if self.chmodels['rv']['type']=='0D_elast' or self.chmodels['rv']['type']=='prescribed' or self.chmodels['rv']['type']=='0D_elast_prescr': self.auxmap['V_v_r'] = nc+1
        if self.chmodels['la']['type']=='0D_elast' or self.chmodels['la']['type']=='prescribed' or self.chmodels['la']['type']=='0D_elast_prescr': self.auxmap['V_at_l'] = nc+2
        if self.chmodels['ra']['type']=='0D_elast' or self.chmodels['ra']['type']=='prescribed' or self.chmodels['ra']['type']=='0D_elast_prescr': self.auxmap['V_at_r'] = nc+3
        self.auxmap['V_ar_sys'] = nc+4
        self.auxmap['V_arperi_sys'] = nc+5
        self.auxmap['V_venspl_sys'] = nc+6
        self.auxmap['V_venespl_sys'] = nc+7
        self.auxmap['V_venmsc_sys'] = nc+8
        self.auxmap['V_vencer_sys'] = nc+9
        self.auxmap['V_ven_sys'] = nc+10
        self.auxmap['V_ar_pul'] = nc+11
        self.auxmap['V_cap_pul'] = nc+12
        self.auxmap['V_ven_pul'] = nc+13


        # populate auxiliary variable vector
        for i in range(nc): self.a_[i] = self.c_[i]
        self.a_[nc+0]  = VQ_v_l_ * self.switch_V[0]
        self.a_[nc+1]  = VQ_v_r_ * self.switch_V[1]
        self.a_[nc+2]  = VQ_at_l_ * self.switch_V[2]
        self.a_[nc+3]  = VQ_at_r_ * self.switch_V[3]
        self.a_[nc+4]  = self.C_ar_sys * p_ard_sys_ + self.V_ar_sys_u
        self.a_[nc+5]  = (self.C_arspl_sys+self.C_arespl_sys+self.C_armsc_sys+self.C_arcer_sys) * p_arperi_sys_ + self.V_arspl_sys_u+self.V_arespl_sys_u+self.V_armsc_sys_u+self.V_arcer_sys_u
        self.a_[nc+6]  = self.C_venspl_sys * p_venspl_sys_ + self.V_venspl_sys_u
        self.a_[nc+7]  = self.C_venespl_sys * p_venespl_sys_ + self.V_venespl_sys_u
        self.a_[nc+8]  = self.C_venmsc_sys * p_venmsc_sys_ + self.V_venmsc_sys_u
        self.a_[nc+9]  = self.C_vencer_sys * p_vencer_sys_ + self.V_vencer_sys_u
        self.a_[nc+10] = self.C_ven_sys * p_ven_sys_ + self.V_ven_sys_u
        self.a_[nc+11] = self.C_ar_pul * p_ar_pul_ + self.V_ar_pul_u
        self.a_[nc+12] = self.C_cap_pul * p_cap_pul_ + self.V_cap_pul_u
        self.a_[nc+13] = self.C_ven_pul * p_ven_pul_ + self.V_ven_pul_u


    def initialize(self, var, iniparam):

        var[self.varmap['q_vin_l']]                           = iniparam['q_vin_l_0']
        var[self.varmap[self.vname[2]]]                       = iniparam[self.vname[2]+'_0']
        var[self.varmap['q_vout_l']]                          = iniparam['q_vout_l_0']
        var[self.varmap[self.vname[0]]]                       = iniparam[self.vname[0]+'_0']
        var[self.varmap['p_ar_sys']]                          = iniparam['p_ar_sys_0']
        try: var[self.varmap['q_arp_sys']]                    = iniparam['q_arp_sys_0']
        except: var[self.varmap['q_arp_sys']]                 = iniparam['q_ar_sys_0']
        try: var[self.varmap['p_ard_sys']]                    = iniparam['p_ard_sys_0']
        except: var[self.varmap['p_ard_sys']]                 = iniparam['p_ar_sys_0']
        var[self.varmap['q_ar_sys']]                          = iniparam['q_ar_sys_0']
        var[self.varmap['p_arperi_sys']]                      = iniparam['p_arperi_sys_0']
        var[self.varmap['q_arspl_sys']]                       = iniparam['q_arspl_sys_0']
        var[self.varmap['q_arespl_sys']]                      = iniparam['q_arespl_sys_0']
        var[self.varmap['q_armsc_sys']]                       = iniparam['q_armsc_sys_0']
        var[self.varmap['q_arcer_sys']]                       = iniparam['q_arcer_sys_0']
        var[self.varmap['p_venspl_sys']]                      = iniparam['p_venspl_sys_0']
        var[self.varmap['q_venspl_sys']]                      = iniparam['q_venspl_sys_0']
        var[self.varmap['p_venespl_sys']]                     = iniparam['p_venespl_sys_0']
        var[self.varmap['q_venespl_sys']]                     = iniparam['q_venespl_sys_0']
        var[self.varmap['p_venmsc_sys']]                      = iniparam['p_venmsc_sys_0']
        var[self.varmap['q_venmsc_sys']]                      = iniparam['q_venmsc_sys_0']
        var[self.varmap['p_vencer_sys']]                      = iniparam['p_vencer_sys_0']
        var[self.varmap['q_vencer_sys']]                      = iniparam['q_vencer_sys_0']
        var[self.varmap['p_ven_sys']]                         = iniparam['p_ven_sys_0']
        for n in range(self.vs):
            try: var[self.varmap['q_ven'+str(n+1)+'_sys']]    = iniparam['q_ven'+str(n+1)+'_sys_0']
            except: var[self.varmap['q_ven'+str(n+1)+'_sys']]    = iniparam['q_ven_sys_0']
        var[self.varmap['q_vin_r']]                           = iniparam['q_vin_r_0']
        var[self.varmap[self.vname[3]]]                       = iniparam[self.vname[3]+'_0']
        var[self.varmap['q_vout_r']]                          = iniparam['q_vout_r_0']
        var[self.varmap[self.vname[1]]]                       = iniparam[self.vname[1]+'_0']
        var[self.varmap['p_ar_pul']]                          = iniparam['p_ar_pul_0']
        var[self.varmap['q_ar_pul']]                          = iniparam['q_ar_pul_0']
        var[self.varmap['p_cap_pul']]                         = iniparam['p_cap_pul_0']
        var[self.varmap['q_cap_pul']]                         = iniparam['q_cap_pul_0']
        var[self.varmap['p_ven_pul']]                         = iniparam['p_ven_pul_0']
        for n in range(self.vp):
            try: var[self.varmap['q_ven'+str(n+1)+'_pul']]    = iniparam['q_ven'+str(n+1)+'_pul_0']
            except: var[self.varmap['q_ven'+str(n+1)+'_pul']]    = iniparam['q_ven_pul_0']

        if self.cormodel is not None:
            self.corcirc.initialize(var, iniparam)


    def check_periodic(self, varTc, varTc_old, auxTc, auxTc_old, eps, check, cyclerr):

        if isinstance(varTc, np.ndarray): varTc_sq, varTc_old_sq = varTc, varTc_old
        else: varTc_sq, varTc_old_sq = allgather_vec(varTc, self.comm), allgather_vec(varTc_old, self.comm)

        vals = []

        if check[0]=='allvar':

            var_ids, aux_ids = list(self.varmap.values()), []

        elif check[0]=='allvaraux':

            var_ids, aux_ids = list(self.varmap.values()), list(self.auxmap.values())

        elif check[0]=='pQvar':

            var_ids, aux_ids = [self.varmap[self.vname[2]],self.varmap[self.vname[0]],self.varmap['p_ar_sys'],self.varmap['p_ard_sys'],self.varmap['p_arperi_sys'],self.varmap['p_venspl_sys'],self.varmap['p_venespl_sys'],self.varmap['p_venmsc_sys'],self.varmap['p_vencer_sys'],self.varmap['p_ven_sys'],self.varmap[self.vname[3]],self.varmap[self.vname[1]],self.varmap['p_ar_pul'],self.varmap['p_cap_pul'],self.varmap['p_ven_pul']], []

        elif check[0]=='specific':

            var_ids = []
            for k in range(len(self.varmap)):
                if list(self.varmap.keys())[k] in check[1]:
                    var_ids.append(list(self.varmap.values())[k])

            aux_ids = []
            for k in range(len(self.auxmap)):
                if list(self.auxmap.keys())[k] in check[1]:
                    aux_ids.append(list(self.auxmap.values())[k])

        else:
            raise NameError("Unknown check option!")

        # compute the errors
        for i in range(len(varTc_sq)):
            if i in var_ids:
                vals.append( math.fabs((varTc_sq[i]-varTc_old_sq[i])/max(1.0,math.fabs(varTc_old_sq[i]))) )

        for i in range(len(auxTc)):
            if i in aux_ids:
                vals.append( math.fabs((auxTc[i]-auxTc_old[i])/max(1.0,math.fabs(auxTc_old[i]))) )

        cyclerr[0] = max(vals)

        if cyclerr[0] <= eps:
            is_periodic = True
        else:
            is_periodic = False

        if not isinstance(varTc, np.ndarray): del varTc_sq, varTc_old_sq

        return is_periodic


    def print_to_screen(self, var, aux):

        if isinstance(var, np.ndarray): var_sq = var
        else: var_sq = allgather_vec(var, self.comm)

        nc = len(self.c_)

        if self.comm.rank == 0:

            print("Output of 0D vascular model (syspulcapcor):")

            for i in range(nc):
                print('{:<12s}{:<3s}{:<10.3f}'.format(self.cname[i],' = ',aux[self.auxmap[self.cname[i]]]))

            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format(self.vname[2],' = ',var_sq[self.varmap[self.vname[2]]],'   ',self.vname[3],' = ',var_sq[self.varmap[self.vname[3]]]))
            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format(self.vname[0],' = ',var_sq[self.varmap[self.vname[0]]],'   ',self.vname[1],' = ',var_sq[self.varmap[self.vname[1]]]))

            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format('p_ar_sys',' = ',var_sq[self.varmap['p_ar_sys']],'   ','p_ar_pul',' = ',var_sq[self.varmap['p_ar_pul']]))
            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format('p_arperi_sys',' = ',var_sq[self.varmap['p_arperi_sys']],'   ','p_cap_pul',' = ',var_sq[self.varmap['p_cap_pul']]))
            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}'.format('p_ven_sys',' = ',var_sq[self.varmap['p_ven_sys']],'   ','p_ven_pul',' = ',var_sq[self.varmap['p_ven_pul']]))

            sys.stdout.flush()

        if not isinstance(var, np.ndarray): del var_sq




def postprocess_groups_syspulcapcor(groups, coronarymodel=None, indpertaftercyl=0, multiscalegandr=False):

    # index 0
    groups.append({'pres_time_sys_l'  : ['p_at_l', 'p_v_l', 'p_ar_sys', 'p_arperi_sys', 'p_venspl_sys', 'p_venespl_sys', 'p_venmsc_sys', 'p_vencer_sys', 'p_vencor_sys', 'p_ven_sys'],
                'tex'                 : ['$p_{\\\mathrm{at}}^{\\\ell}$', '$p_{\\\mathrm{v}}^{\\\ell}$', '$p_{\\\mathrm{ar}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ar,peri}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ven,spl}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ven,espl}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ven,msc}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ven,cer}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ven,cor}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{ven}}^{\\\mathrm{sys}}$'],
                'lines'               : [1, 2, 3, 4, 10, 11, 12, 13, 14, 15]})
    # index 1
    groups.append({'pres_time_pul_r'  : ['p_at_r', 'p_v_r', 'p_ar_pul', 'p_cap_pul', 'p_ven_pul'],
                'tex'                 : ['$p_{\\\mathrm{at}}^{r}$', '$p_{\\\mathrm{v}}^{r}$', '$p_{\\\mathrm{ar}}^{\\\mathrm{pul}}$', '$p_{\\\mathrm{cap}}^{\\\mathrm{pul}}$', '$p_{\\\mathrm{ven}}^{\\\mathrm{pul}}$'],
                'lines'               : [16, 17, 18, 19, 20]})
    # index 2
    groups.append({'flux_time_sys_l'  : ['q_vin_l', 'q_vout_l', 'q_ar_sys', 'q_arspl_sys', 'q_arespl_sys', 'q_armsc_sys', 'q_arcer_sys', 'q_arcor_sys', 'q_venspl_sys', 'q_venespl_sys', 'q_venmsc_sys', 'q_vencer_sys', 'q_ven1_sys', 'q_ven2_sys'],
                'tex'                 : ['$q_{\\\mathrm{v,in}}^{\\\ell}$', '$q_{\\\mathrm{v,out}}^{\\\ell}$', '$q_{\\\mathrm{ar}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ar,spl}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ar,espl}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ar,msc}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ar,cer}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ar,cor}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ven,spl}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ven,espl}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ven,msc}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ven,cer}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ven,1}}^{\\\mathrm{sys}}$', '$q_{\\\mathrm{ven,2}}^{\\\mathrm{sys}}$'],
                'lines'               : [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]})
    # index 3
    groups.append({'flux_time_pul_r'  : ['q_vin_r', 'q_vout_r', 'q_ar_pul', 'q_cap_pul', 'q_ven1_pul', 'q_ven2_pul', 'q_ven3_pul', 'q_ven4_pul'],
                'tex'                 : ['$q_{\\\mathrm{v,in}}^{r}$', '$q_{\\\mathrm{v,out}}^{r}$', '$q_{\\\mathrm{ar}}^{\\\mathrm{pul}}$', '$q_{\\\mathrm{cap}}^{\\\mathrm{pul}}$', '$q_{\\\mathrm{ven,1}}^{\\\mathrm{pul}}$', '$q_{\\\mathrm{ven,2}}^{\\\mathrm{pul}}$', '$q_{\\\mathrm{ven,3}}^{\\\mathrm{pul}}$', '$q_{\\\mathrm{ven,4}}^{\\\mathrm{pul}}$'],
                'lines'               : [16, 17, 18, 19, 20, 201, 202, 203]})
    # index 4
    groups.append({'vol_time_l_r'     : ['V_at_l', 'V_v_l', 'V_at_r', 'V_v_r'],
                'tex'                 : ['$V_{\\\mathrm{at}}^{\\\ell}$', '$V_{\\\mathrm{v}}^{\\\ell}$', '$V_{\\\mathrm{at}}^{r}$', '$V_{\\\mathrm{v}}^{r}$'],
                'lines'               : [1, 2, 16, 17]})
    # index 5
    groups.append({'vol_time_compart' : ['V_at_l', 'V_v_l', 'V_at_r', 'V_v_r', 'V_ar_sys', 'V_arcor_sys', 'V_arperi_sys', 'V_venspl_sys', 'V_venespl_sys', 'V_venmsc_sys', 'V_vencer_sys', 'V_vencor_sys', 'V_ven_sys', 'V_ar_pul', 'V_cap_pul', 'V_ven_pul'],
                'tex'                 : ['$V_{\\\mathrm{at}}^{\\\ell}$', '$V_{\\\mathrm{v}}^{\\\ell}$', '$V_{\\\mathrm{at}}^{r}$', '$V_{\\\mathrm{v}}^{r}$', '$V_{\\\mathrm{ar}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ar,cor}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ar,peri}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven,spl}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven,espl}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven,msc}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven,cer}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven,cor}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ven}}^{\\\mathrm{sys}}$', '$V_{\\\mathrm{ar}}^{\\\mathrm{pul}}$', '$V_{\\\mathrm{cap}}^{\\\mathrm{pul}}$', '$V_{\\\mathrm{ven}}^{\\\mathrm{pul}}$'],
                'lines'               : [1, 2, 16, 17, 3, 9, 4, 10, 11, 12, 13, 14, 15, 18, 19, 20]})

    # all volumes summed up for conservation check
    groups[5]['vol_time_compart'].append('V_all')
    groups[5]['tex'].append('$\\\sum V$')
    groups[5]['lines'].append(99)

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
