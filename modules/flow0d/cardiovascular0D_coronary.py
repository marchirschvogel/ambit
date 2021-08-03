#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import sympy as sp


# simplest form or coronary model: a 2-element Windkessel for the whole coronary compartment
#\begin{align}
#&C_{\mathrm{cor}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{cor,in}}^{\mathrm{sys}} - q_{\mathrm{cor}}^{\mathrm{sys}}\\
#&R_{\mathrm{cor}}^{\mathrm{sys}}\,q_{\mathrm{cor}}^{\mathrm{sys}}=p_{\mathrm{ar}}^{\mathrm{sys}}-p_{\mathrm{at}}^{r}
#\end{align}

class coronary_circ_RC():
    
    def __init__(self, params, varmap, auxmap, vs):
        
        self.R_cor_sys = params['R_cor_sys']
        self.C_cor_sys = params['C_cor_sys']
        
        try: self.V_cor_sys_u = params['V_cor_sys_u']
        except: self.V_cor_sys_u = 0
        
        self.ndcor = 2
        
        self.varmap = varmap
        self.auxmap = auxmap

        self.vs = vs
    

    def equation_map(self, vindex, aindex, x_, a_, df_, f_, p_ar_, p_at_):
        
        self.varmap['q_cor_sys_in']                = vindex
        self.varmap['q_ven'+str(self.vs+1)+'_sys'] = vindex+1

        q_cor_sys_in_ = sp.Symbol('q_cor_sys_in_')
        q_cor_sys_    = sp.Symbol('q_ven'+str(self.vs+1)+'_sys_')
        
        x_[self.varmap['q_cor_sys_in']]                = q_cor_sys_in_
        x_[self.varmap['q_ven'+str(self.vs+1)+'_sys']] = q_cor_sys_

        # populate df_ and f_ arrays
        df_[vindex]   = self.C_cor_sys * p_ar_[0]                     # coronary volume rate
        df_[vindex+1] = 0.
        
        f_[vindex]   = q_cor_sys_ - q_cor_sys_in_                     # coronary flow balance
        f_[vindex+1] = (p_at_ - p_ar_[0])/self.R_cor_sys + q_cor_sys_ # coronary momentum

        
        # auxiliary map and variables
        self.auxmap['V_cor_sys'] = aindex
        
        a_[self.auxmap['V_cor_sys']] = self.C_cor_sys * p_ar_[0] + self.V_cor_sys_u
        
        return [q_cor_sys_in_], q_cor_sys_


    def initialize(self, var, iniparam):
        
        try: var[self.varmap['q_cor_sys_in']]                   = iniparam['q_cor_sys_in_0']
        except: var[self.varmap['q_cor_sys_in']]                   = iniparam['q_cor_sys_0']
        try: var[self.varmap['q_ven'+str(self.vs+1)+'_sys']]    = iniparam['q_ven'+str(self.vs+1)+'_sys_0']
        except: var[self.varmap['q_ven'+str(self.vs+1)+'_sys']]    = iniparam['q_ven_sys_0']




# two 2-element Windkessel models in series for the arterial and the venous coronary compartment
#\begin{align}
#&C_{\mathrm{ar,cor}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar,cor,in}}^{\mathrm{sys}} - q_{\mathrm{ar,cor}}^{\mathrm{sys}}\\
#&R_{\mathrm{ar,cor}}^{\mathrm{sys}}\,q_{\mathrm{ar,cor}}^{\mathrm{sys}}=p_{\mathrm{ar}}^{\mathrm{sys}}-p_{\mathrm{ven,cor}}^{\mathrm{sys}}\\
#&C_{\mathrm{ven,cor}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven,cor}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar,cor}}^{\mathrm{sys}} - q_{\mathrm{ven,cor}}^{\mathrm{sys}}\\
#&R_{\mathrm{ven,cor}}^{\mathrm{sys}}\,q_{\mathrm{ven,cor}}^{\mathrm{sys}}=p_{\mathrm{ven,cor}}^{\mathrm{sys}}-p_{\mathrm{at}}^{r}
#\end{align}

class coronary_circ_RCar_RCven():
    
    def __init__(self, params, varmap, auxmap, vs):
        
        self.R_arcor_sys = params['R_arcor_sys']
        self.C_arcor_sys = params['C_arcor_sys']
        
        self.R_vencor_sys = params['R_vencor_sys']
        self.C_vencor_sys = params['C_vencor_sys']
        
        try: self.V_arcor_sys_u = params['V_arcor_sys_u']
        except: self.V_arcor_sys_u = 0
        try: self.V_vencor_sys_u = params['V_vencor_sys_u']
        except: self.V_vencor_sys_u = 0
        
        self.ndcor = 4
        
        self.varmap = varmap
        self.auxmap = auxmap

        self.vs = vs
    

    def equation_map(self, vindex, aindex, x_, a_, df_, f_, p_ar_, p_at_):
        
        self.varmap['q_arcor_sys_in']              = vindex
        self.varmap['q_arcor_sys']                 = vindex+1
        self.varmap['p_vencor_sys']                = vindex+2
        self.varmap['q_ven'+str(self.vs+1)+'_sys'] = vindex+3

        q_arcor_sys_in_   = sp.Symbol('q_arcor_sys_in_')
        q_arcor_sys_      = sp.Symbol('q_arcor_sys_')
        p_vencor_sys_     = sp.Symbol('p_vencor_sys_')
        q_vencor_sys_out_ = sp.Symbol('q_ven'+str(self.vs+1)+'_sys_')
        
        x_[self.varmap['q_arcor_sys_in']]              = q_arcor_sys_in_
        x_[self.varmap['q_arcor_sys']]                 = q_arcor_sys_
        x_[self.varmap['p_vencor_sys']]                = p_vencor_sys_
        x_[self.varmap['q_ven'+str(self.vs+1)+'_sys']] = q_vencor_sys_out_

        # populate df_ and f_ arrays
        df_[vindex]   = self.C_arcor_sys * p_ar_[0]                                  # coronary arterial volume rate
        df_[vindex+1] = 0.
        df_[vindex+2] = self.C_vencor_sys * p_vencor_sys_                            # coronary venous/coronary sinus volume rate
        df_[vindex+3] = 0
        
        f_[vindex]   = q_arcor_sys_ - q_arcor_sys_in_                                # coronary arterial flow balance
        f_[vindex+1] = (p_vencor_sys_ - p_ar_[0])/self.R_arcor_sys + q_arcor_sys_    # coronary arterial momentum
        f_[vindex+2] = q_vencor_sys_out_ - q_arcor_sys_                              # coronary venous/coronary sinus flow balance
        f_[vindex+3] = (p_at_ - p_vencor_sys_)/self.R_vencor_sys + q_vencor_sys_out_ # coronary venous/coronary sinus momentum

        
        # auxiliary map and variables
        self.auxmap['V_arcor_sys']  = aindex
        self.auxmap['V_vencor_sys'] = aindex+1
        
        a_[self.auxmap['V_arcor_sys']]  = self.C_arcor_sys * p_ar_[0] + self.V_arcor_sys_u
        a_[self.auxmap['V_vencor_sys']] = self.C_vencor_sys * p_vencor_sys_ + self.V_vencor_sys_u
        
        return [q_arcor_sys_in_], q_vencor_sys_out_


    def initialize(self, var, iniparam):
        
        try: var[self.varmap['q_arcor_sys_in']]                 = iniparam['q_arcor_sys_in_0']
        except: var[self.varmap['q_arcor_sys_in']]                 = iniparam['q_arcor_sys_0']
        var[self.varmap['q_arcor_sys']]                         = iniparam['q_arcor_sys_0']
        
        var[self.varmap['p_vencor_sys']]                        = iniparam['p_vencor_sys_0']
        
        try: var[self.varmap['q_ven'+str(self.vs+1)+'_sys']]    = iniparam['q_ven'+str(self.vs+1)+'_sys_0']
        except: var[self.varmap['q_ven'+str(self.vs+1)+'_sys']]    = iniparam['q_ven_sys_0']





# left and right heart coronary model: two LRC Windkessel models in parallel, one for left and other for right coronary compartment
#\begin{align}
#&C_{\mathrm{cor}}^{\mathrm{sys},\ell} \frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{cor,in}}^{\mathrm{sys},\ell} - q_{\mathrm{cor}}^{\mathrm{sys},\ell}\\
#&L_{\mathrm{cor}}^{\mathrm{sys},\ell}\frac{\mathrm{d}q_{\mathrm{cor}}^{\mathrm{sys},\ell}}{\mathrm{d}t} + R_{\mathrm{cor}}^{\mathrm{sys},\ell}\,q_{\mathrm{cor}}^{\mathrm{sys},\ell}=p_{\mathrm{ar}}^{\mathrm{sys}}-p_{\mathrm{at}}^{r}\\
#&C_{\mathrm{cor}}^{\mathrm{sys},r} \frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{cor,in}}^{\mathrm{sys},r} - q_{\mathrm{cor}}^{\mathrm{sys},r}\\
#&L_{\mathrm{cor}}^{\mathrm{sys},r}\frac{\mathrm{d}q_{\mathrm{cor}}^{\mathrm{sys},r}}{\mathrm{d}t} + R_{\mathrm{cor}}^{\mathrm{sys},r}\,q_{\mathrm{cor}}^{\mathrm{sys},r}=p_{\mathrm{ar}}^{\mathrm{sys}}-p_{\mathrm{at}}^{r}\\
#&0=q_{\mathrm{cor}}^{\mathrm{sys},\ell}+q_{\mathrm{cor}}^{\mathrm{sys},r}-q_{\mathrm{cor}}^{\mathrm{sys}}
#\end{align}

class coronary_circ_RLCl_RLCr():
    
    def __init__(self, params, varmap, auxmap, vs):
        
        self.R_cor_sys_l = params['R_cor_sys_l']
        self.L_cor_sys_l = params['L_cor_sys_l']
        self.C_cor_sys_l = params['C_cor_sys_l']

        self.R_cor_sys_r = params['R_cor_sys_r']
        self.L_cor_sys_r = params['L_cor_sys_r']
        self.C_cor_sys_r = params['C_cor_sys_r']
        
        try: self.V_cor_sys_l_u = params['V_cor_sys_l_u']
        except: self.V_cor_sys_l_u = 0
        try: self.V_cor_sys_r_u = params['V_cor_sys_r_u']
        except: self.V_cor_sys_r_u = 0
        
        self.ndcor = 5
        
        self.varmap = varmap
        self.auxmap = auxmap

        self.vs = vs
    

    def equation_map(self, vindex, aindex, x_, a_, df_, f_, p_ar_, p_at_):
        
        self.varmap['q_cor_sys_l_in']              = vindex
        self.varmap['q_cor_sys_l']                 = vindex+1
        self.varmap['q_cor_sys_r_in']              = vindex+2
        self.varmap['q_cor_sys_r']                 = vindex+3
        self.varmap['q_ven'+str(self.vs+1)+'_sys'] = vindex+4

        q_cor_sys_l_in_ = sp.Symbol('q_cor_sys_l_in_')
        q_cor_sys_l_    = sp.Symbol('q_cor_sys_l_')
        q_cor_sys_r_in_ = sp.Symbol('q_cor_sys_r_in_')
        q_cor_sys_r_    = sp.Symbol('q_cor_sys_r_')        
        q_cor_sys_      = sp.Symbol('q_ven'+str(self.vs+1)+'_sys_')
        
        x_[self.varmap['q_cor_sys_l_in']]              = q_cor_sys_l_in_
        x_[self.varmap['q_cor_sys_l']]                 = q_cor_sys_l_
        x_[self.varmap['q_cor_sys_r_in']]              = q_cor_sys_r_in_
        x_[self.varmap['q_cor_sys_r']]                 = q_cor_sys_r_
        x_[self.varmap['q_ven'+str(self.vs+1)+'_sys']] = q_cor_sys_

        # populate df_ and f_ arrays
        df_[vindex]   = self.C_cor_sys_l * p_ar_[0]                        # left coronary volume rate
        df_[vindex+1] = (self.L_cor_sys_l/self.R_cor_sys_l) * q_cor_sys_l_ # left coronary inertia
        df_[vindex+2] = self.C_cor_sys_r * p_ar_[1]                        # right coronary volume rate
        df_[vindex+3] = (self.L_cor_sys_r/self.R_cor_sys_r) * q_cor_sys_r_ # right coronary inertia
        df_[vindex+4] = 0.
        
        f_[vindex]   = q_cor_sys_l_ - q_cor_sys_l_in_                      # left coronary flow balance
        f_[vindex+1] = (p_at_ - p_ar_[0])/self.R_cor_sys_l + q_cor_sys_l_  # left coronary momentum
        f_[vindex+2] = q_cor_sys_r_ - q_cor_sys_r_in_                      # right coronary flow balance
        f_[vindex+3] = (p_at_ - p_ar_[1])/self.R_cor_sys_r + q_cor_sys_r_  # right coronary momentum
        f_[vindex+4] = q_cor_sys_ - q_cor_sys_l_ - q_cor_sys_r_            # coronary sinus flow balance

        # auxiliary map and variables
        self.auxmap['V_cor_sys_l'] = aindex
        self.auxmap['V_cor_sys_r'] = aindex+1
        
        a_[self.auxmap['V_cor_sys_l']] = self.C_cor_sys_l * p_ar_[0] + self.V_cor_sys_l_u
        a_[self.auxmap['V_cor_sys_r']] = self.C_cor_sys_r * p_ar_[1] + self.V_cor_sys_r_u
        
        return [q_cor_sys_l_in_,q_cor_sys_r_in_], q_cor_sys_


    def initialize(self, var, iniparam):
        
        try: var[self.varmap['q_cor_sys_l_in']]                 = iniparam['q_cor_sys_l_in_0']
        except: var[self.varmap['q_cor_sys_l_in']]                 = iniparam['q_cor_sys_0']

        try: var[self.varmap['q_cor_sys_l']]                    = iniparam['q_cor_sys_l_0']
        except: var[self.varmap['q_cor_sys_l']]                    = iniparam['q_cor_sys_0']

        try: var[self.varmap['q_cor_sys_r_in']]                 = iniparam['q_cor_sys_r_in_0']
        except: var[self.varmap['q_cor_sys_r_in']]                 = iniparam['q_cor_sys_0']

        try: var[self.varmap['q_cor_sys_r']]                    = iniparam['q_cor_sys_r_0']
        except: var[self.varmap['q_cor_sys_r']]                    = iniparam['q_cor_sys_0']
        
        try: var[self.varmap['q_ven'+str(self.vs+1)+'_sys']]    = iniparam['q_ven'+str(self.vs+1)+'_sys_0']
        except: var[self.varmap['q_ven'+str(self.vs+1)+'_sys']]    = iniparam['q_ven_sys_0']
