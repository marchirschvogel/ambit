#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import sympy as sp

# coronary model with a 3-element Windkessel (ZCR, proximal part) in series with a 2-element Windkessel (CR, distal part)
# according to Vieira et al. (2018) "Patient-specific modeling of right coronary circulation vulnerability post-liver transplant in Alagille’s syndrome", PLoS ONE 13(11), e0205829
# here their R_e is Z_corp_sys, C_e is C_corp_sys, R_p is R_corp_sys, C_i is C_cord_sys, and R_d is R_cord_sys
# the distal compliance is fed by the left ventricular pressure in order to have a phase-dependent tone of the coronary
# (coronaries almost entirely fill in diastole, not during systole)
#\begin{align}
#&C_{\mathrm{cor,p}}^{\mathrm{sys}} \left(\frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t}-Z_{\mathrm{cor,p}}^{\mathrm{sys}}\frac{\mathrm{d}q_{\mathrm{cor,p,in}}^{\mathrm{sys}}}{\mathrm{d}t}\right) = q_{\mathrm{cor,p,in}}^{\mathrm{sys}} - q_{\mathrm{cor,p}}^{\mathrm{sys}}\\
#&R_{\mathrm{cor,p}}^{\mathrm{sys}}\,q_{\mathrm{cor,p}}^{\mathrm{sys}}=p_{\mathrm{ar}}^{\mathrm{sys}}-p_{\mathrm{cor,d}}^{\mathrm{sys}} - Z_{\mathrm{cor,p}}^{\mathrm{sys}}\,q_{\mathrm{cor,p,in}}^{\mathrm{sys}}\\
#&C_{\mathrm{cor,d}}^{\mathrm{sys}} \frac{\mathrm{d}(p_{\mathrm{cor,d}}^{\mathrm{sys}}-p_{\mathrm{v}}^{\ell})}{\mathrm{d}t} = q_{\mathrm{cor,p}}^{\mathrm{sys}} - q_{\mathrm{cor,d}}^{\mathrm{sys}}\\
#&R_{\mathrm{cor,d}}^{\mathrm{sys}}\,q_{\mathrm{cor,d}}^{\mathrm{sys}}=p_{\mathrm{cor,d}}^{\mathrm{sys}}-p_{\mathrm{at}}^{r}
#\end{align}

class coronary_circ_ZCRp_CRd():

    def __init__(self, params, varmap, auxmap, vs):

        self.Z_corp_sys = params['Z_corp_sys']
        self.C_corp_sys = params['C_corp_sys']
        self.R_corp_sys = params['R_corp_sys']
        self.C_cord_sys = params['C_cord_sys']
        self.R_cord_sys = params['R_cord_sys']

        try: self.V_corp_sys_u = params['V_corp_sys_u']
        except: self.V_corp_sys_u = 0
        try: self.V_cord_sys_u = params['V_cord_sys_u']
        except: self.V_cord_sys_u = 0

        self.ndcor = 4

        self.varmap = varmap
        self.auxmap = auxmap

        self.vs = vs


    def equation_map(self, vindex, aindex, x_, a_, df_, f_, p_ar_, p_v_, p_at_):

        self.varmap['q_corp_sys_in']               = vindex
        self.varmap['q_corp_sys']                  = vindex+1
        self.varmap['p_cord_sys']                  = vindex+2
        self.varmap['q_ven'+str(self.vs+1)+'_sys'] = vindex+3

        q_corp_sys_in_ = sp.Symbol('q_corp_sys_in_')
        q_corp_sys_    = sp.Symbol('q_corp_sys_')
        p_cord_sys_    = sp.Symbol('p_cord_sys_')
        q_cord_sys_    = sp.Symbol('q_ven'+str(self.vs+1)+'_sys_')

        x_[self.varmap['q_corp_sys_in']]               = q_corp_sys_in_
        x_[self.varmap['q_corp_sys']]                  = q_corp_sys_
        x_[self.varmap['p_cord_sys']]                  = p_cord_sys_
        x_[self.varmap['q_ven'+str(self.vs+1)+'_sys']] = q_cord_sys_

        # populate df_ and f_ arrays
        df_[vindex]   = self.C_corp_sys * (p_ar_[0] - self.Z_corp_sys * q_corp_sys_in_) # coronary proximal volume rate
        df_[vindex+1] = 0.
        df_[vindex+2] = self.C_cord_sys * (p_cord_sys_ - p_v_)                          # coronary distal volume rate
        df_[vindex+3] = 0.

        f_[vindex]   = q_corp_sys_ - q_corp_sys_in_                                                              # coronary proximal flow balance
        f_[vindex+1] = (p_cord_sys_ - p_ar_[0] + self.Z_corp_sys * q_corp_sys_in_)/self.R_corp_sys + q_corp_sys_ # coronary proximal momentum
        f_[vindex+2] = q_cord_sys_ - q_corp_sys_                                                                 # coronary distal flow balance
        f_[vindex+3] = (p_at_ - p_cord_sys_)/self.R_cord_sys + q_cord_sys_                                       # coronary distal momentum

        # auxiliary map and variables
        self.auxmap['V_corp_sys'] = aindex
        self.auxmap['V_cord_sys'] = aindex+1

        a_[self.auxmap['V_corp_sys']] = self.C_corp_sys * (p_ar_[0] - self.Z_corp_sys * q_corp_sys_in_) + self.V_corp_sys_u
        a_[self.auxmap['V_cord_sys']] = self.C_cord_sys * (p_cord_sys_ - p_v_) + self.V_cord_sys_u

        # safety check that we don't hand in a zero symbol for p_v
        if p_v_ is sp.S.Zero: raise ValueError("Zero symbol for left ventricular pressure!")

        return [q_corp_sys_in_], q_cord_sys_


    def initialize(self, var, iniparam):

        try: var[self.varmap['q_corp_sys_in']]                  = iniparam['q_corp_sys_in_0']
        except: var[self.varmap['q_corp_sys_in']]                  = iniparam['q_corp_sys_0']
        var[self.varmap['q_corp_sys']]                          = iniparam['q_corp_sys_0']
        var[self.varmap['p_cord_sys']]                          = iniparam['p_cord_sys_0']

        try: var[self.varmap['q_ven'+str(self.vs+1)+'_sys']]    = iniparam['q_ven'+str(self.vs+1)+'_sys_0']
        except: var[self.varmap['q_ven'+str(self.vs+1)+'_sys']]    = iniparam['q_ven_sys_0']


    def print_to_screen(self, var_sq, aux):

        print("Output of 0D coronary model (ZCRp_CRd):")

        print('{:<10s}{:<3s}{:<7.3f}'.format('p_cord_sys',' = ',var_sq[self.varmap['p_cord_sys']]))
        sys.stdout.flush()


# equivalent model to ZCRp_CRd, but individually for left and right coronary arteries
#\begin{align}
#&C_{\mathrm{cor,p}}^{\mathrm{sys},\ell} \left(\frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{sys},\ell}}{\mathrm{d}t}-Z_{\mathrm{cor,p}}^{\mathrm{sys},\ell}\frac{\mathrm{d}q_{\mathrm{cor,p,in}}^{\mathrm{sys},\ell}}{\mathrm{d}t}\right) = q_{\mathrm{cor,p,in}}^{\mathrm{sys},\ell} - q_{\mathrm{cor,p}}^{\mathrm{sys},\ell}\\
#&R_{\mathrm{cor,p}}^{\mathrm{sys},\ell}\,q_{\mathrm{cor,p}}^{\mathrm{sys},\ell}=p_{\mathrm{ar}}^{\mathrm{sys}}-p_{\mathrm{cor,d}}^{\mathrm{sys},\ell} - Z_{\mathrm{cor,p}}^{\mathrm{sys},\ell}\,q_{\mathrm{cor,p,in}}^{\mathrm{sys},\ell}\\
#&C_{\mathrm{cor,d}}^{\mathrm{sys},\ell} \frac{\mathrm{d}(p_{\mathrm{cor,d}}^{\mathrm{sys},\ell}-p_{\mathrm{v}}^{\ell})}{\mathrm{d}t} = q_{\mathrm{cor,p}}^{\mathrm{sys},\ell} - q_{\mathrm{cor,d}}^{\mathrm{sys},\ell}\\
#&R_{\mathrm{cor,d}}^{\mathrm{sys},\ell}\,q_{\mathrm{cor,d}}^{\mathrm{sys},\ell}=p_{\mathrm{cor,d}}^{\mathrm{sys},\ell}-p_{\mathrm{at}}^{r}\\
#&C_{\mathrm{cor,p}}^{\mathrm{sys},r} \left(\frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{sys},r}}{\mathrm{d}t}-Z_{\mathrm{cor,p}}^{\mathrm{sys},r}\frac{\mathrm{d}q_{\mathrm{cor,p,in}}^{\mathrm{sys},r}}{\mathrm{d}t}\right) = q_{\mathrm{cor,p,in}}^{\mathrm{sys},r} - q_{\mathrm{cor,p}}^{\mathrm{sys},r}\\
#&R_{\mathrm{cor,p}}^{\mathrm{sys},r}\,q_{\mathrm{cor,p}}^{\mathrm{sys},r}=p_{\mathrm{ar}}^{\mathrm{sys}}-p_{\mathrm{cor,d}}^{\mathrm{sys},r} - Z_{\mathrm{cor,p}}^{\mathrm{sys},r}\,q_{\mathrm{cor,p,in}}^{\mathrm{sys},r}\\
#&C_{\mathrm{cor,d}}^{\mathrm{sys},r} \frac{\mathrm{d}(p_{\mathrm{cor,d}}^{\mathrm{sys},r}-p_{\mathrm{v}}^{\ell})}{\mathrm{d}t} = q_{\mathrm{cor,p}}^{\mathrm{sys},r} - q_{\mathrm{cor,d}}^{\mathrm{sys},r}\\
#&R_{\mathrm{cor,d}}^{\mathrm{sys},r}\,q_{\mathrm{cor,d}}^{\mathrm{sys},r}=p_{\mathrm{cor,d}}^{\mathrm{sys},r}-p_{\mathrm{at}}^{r}\\
#&0=q_{\mathrm{cor,d}}^{\mathrm{sys},\ell}+q_{\mathrm{cor,d}}^{\mathrm{sys},r}-q_{\mathrm{cor,d,out}}^{\mathrm{sys}}
#\end{align}

class coronary_circ_ZCRp_CRd_lr():

    def __init__(self, params, varmap, auxmap, vs):

        self.Z_corp_sys_l = params['Z_corp_sys_l']
        self.C_corp_sys_l = params['C_corp_sys_l']
        self.R_corp_sys_l = params['R_corp_sys_l']
        self.C_cord_sys_l = params['C_cord_sys_l']
        self.R_cord_sys_l = params['R_cord_sys_l']

        self.Z_corp_sys_r = params['Z_corp_sys_r']
        self.C_corp_sys_r = params['C_corp_sys_r']
        self.R_corp_sys_r = params['R_corp_sys_r']
        self.C_cord_sys_r = params['C_cord_sys_r']
        self.R_cord_sys_r = params['R_cord_sys_r']


        try: self.V_corp_sys_l_u = params['V_corp_sys_l_u']
        except: self.V_corp_sys_l_u = 0
        try: self.V_cord_sys_l_u = params['V_cord_sys_l_u']
        except: self.V_cord_sys_l_u = 0

        try: self.V_corp_sys_r_u = params['V_corp_sys_r_u']
        except: self.V_corp_sys_r_u = 0
        try: self.V_cord_sys_r_u = params['V_cord_sys_r_u']
        except: self.V_cord_sys_r_u = 0

        self.ndcor = 9

        self.varmap = varmap
        self.auxmap = auxmap

        self.vs = vs


    def equation_map(self, vindex, aindex, x_, a_, df_, f_, p_ar_, p_v_, p_at_):

        self.varmap['q_corp_sys_l_in']             = vindex
        self.varmap['q_corp_sys_l']                = vindex+1
        self.varmap['p_cord_sys_l']                = vindex+2
        self.varmap['q_cord_sys_l']                = vindex+3

        self.varmap['q_corp_sys_r_in']             = vindex+4
        self.varmap['q_corp_sys_r']                = vindex+5
        self.varmap['p_cord_sys_r']                = vindex+6
        self.varmap['q_cord_sys_r']                = vindex+7

        self.varmap['q_ven'+str(self.vs+1)+'_sys'] = vindex+8

        q_corp_sys_l_in_ = sp.Symbol('q_corp_sys_l_in_')
        q_corp_sys_l_    = sp.Symbol('q_corp_sys_l_')
        p_cord_sys_l_    = sp.Symbol('p_cord_sys_l_')
        q_cord_sys_l_    = sp.Symbol('q_cord_sys_l_')

        q_corp_sys_r_in_ = sp.Symbol('q_corp_sys_r_in_')
        q_corp_sys_r_    = sp.Symbol('q_corp_sys_r_')
        p_cord_sys_r_    = sp.Symbol('p_cord_sys_r_')
        q_cord_sys_r_    = sp.Symbol('q_cord_sys_r_')

        q_cord_sys_out_ = sp.Symbol('q_ven'+str(self.vs+1)+'_sys_')

        x_[self.varmap['q_corp_sys_l_in']]             = q_corp_sys_l_in_
        x_[self.varmap['q_corp_sys_l']]                = q_corp_sys_l_
        x_[self.varmap['p_cord_sys_l']]                = p_cord_sys_l_
        x_[self.varmap['q_cord_sys_l']]                = q_cord_sys_l_

        x_[self.varmap['q_corp_sys_r_in']]             = q_corp_sys_r_in_
        x_[self.varmap['q_corp_sys_r']]                = q_corp_sys_r_
        x_[self.varmap['p_cord_sys_r']]                = p_cord_sys_r_
        x_[self.varmap['q_cord_sys_r']]                = q_cord_sys_r_

        x_[self.varmap['q_ven'+str(self.vs+1)+'_sys']] = q_cord_sys_out_

        # populate df_ and f_ arrays
        df_[vindex]   = self.C_corp_sys_l * (p_ar_[0] - self.Z_corp_sys_l * q_corp_sys_l_in_) # left coronary proximal volume rate
        df_[vindex+1] = 0.
        df_[vindex+2] = self.C_cord_sys_l * (p_cord_sys_l_ - p_v_)                            # left coronary distal volume rate
        df_[vindex+3] = 0.
        df_[vindex+4] = self.C_corp_sys_r * (p_ar_[1] - self.Z_corp_sys_r * q_corp_sys_r_in_) # right coronary proximal volume rate
        df_[vindex+5] = 0.
        df_[vindex+6] = self.C_cord_sys_r * (p_cord_sys_r_ - p_v_)                            # right coronary distal volume rate
        df_[vindex+7] = 0.
        df_[vindex+8] = 0.

        f_[vindex]   = q_corp_sys_l_ - q_corp_sys_l_in_                                                                    # left coronary proximal flow balance
        f_[vindex+1] = (p_cord_sys_l_ - p_ar_[0] + self.Z_corp_sys_l * q_corp_sys_l_in_)/self.R_corp_sys_l + q_corp_sys_l_ # left coronary proximal momentum
        f_[vindex+2] = q_cord_sys_l_ - q_corp_sys_l_                                                                       # left coronary distal flow balance
        f_[vindex+3] = (p_at_ - p_cord_sys_l_)/self.R_cord_sys_l + q_cord_sys_l_                                           # left coronary distal momentum
        f_[vindex+4] = q_corp_sys_r_ - q_corp_sys_r_in_                                                                    # right coronary proximal flow balance
        f_[vindex+5] = (p_cord_sys_r_ - p_ar_[1] + self.Z_corp_sys_r * q_corp_sys_r_in_)/self.R_corp_sys_r + q_corp_sys_r_ # right coronary proximal momentum
        f_[vindex+6] = q_cord_sys_r_ - q_corp_sys_r_                                                                       # right coronary distal flow balance
        f_[vindex+7] = (p_at_ - p_cord_sys_r_)/self.R_cord_sys_r + q_cord_sys_r_                                           # right coronary distal momentum
        f_[vindex+8] = q_cord_sys_out_ - q_cord_sys_l_ - q_cord_sys_r_                                                     # coronary sinus flow balance

        # auxiliary map and variables
        self.auxmap['V_corp_sys_l'] = aindex
        self.auxmap['V_cord_sys_l'] = aindex+1
        self.auxmap['V_corp_sys_r'] = aindex+2
        self.auxmap['V_cord_sys_r'] = aindex+3

        a_[self.auxmap['V_corp_sys_l']] = self.C_corp_sys_l * (p_ar_[0] - self.Z_corp_sys_l * q_corp_sys_l_in_) + self.V_corp_sys_l_u
        a_[self.auxmap['V_cord_sys_l']] = self.C_cord_sys_l * (p_cord_sys_l_ - p_v_) + self.V_cord_sys_l_u
        a_[self.auxmap['V_corp_sys_r']] = self.C_corp_sys_r * (p_ar_[1] - self.Z_corp_sys_r * q_corp_sys_r_in_) + self.V_corp_sys_r_u
        a_[self.auxmap['V_cord_sys_r']] = self.C_cord_sys_r * (p_cord_sys_r_ - p_v_) + self.V_cord_sys_r_u

        # safety check that we don't hand in a zero symbol for p_v
        if p_v_ is sp.S.Zero: raise ValueError("Zero symbol for left ventricular pressure!")

        return [q_corp_sys_l_in_,q_corp_sys_r_in_], q_cord_sys_out_


    def initialize(self, var, iniparam):

        try: var[self.varmap['q_corp_sys_l_in']]                = iniparam['q_corp_sys_l_in_0']
        except: var[self.varmap['q_corp_sys_l_in']]                = iniparam['q_corp_sys_l_0']
        var[self.varmap['q_corp_sys_l']]                        = iniparam['q_corp_sys_l_0']
        var[self.varmap['p_cord_sys_l']]                        = iniparam['p_cord_sys_l_0']
        var[self.varmap['q_cord_sys_l']]                        = iniparam['q_cord_sys_l_0']

        try: var[self.varmap['q_corp_sys_r_in']]                = iniparam['q_corp_sys_r_in_0']
        except: var[self.varmap['q_corp_sys_r_in']]                = iniparam['q_corp_sys_r_0']
        var[self.varmap['q_corp_sys_r']]                        = iniparam['q_corp_sys_r_0']
        var[self.varmap['p_cord_sys_r']]                        = iniparam['p_cord_sys_r_0']
        var[self.varmap['q_cord_sys_r']]                        = iniparam['q_cord_sys_r_0']

        try: var[self.varmap['q_ven'+str(self.vs+1)+'_sys']]    = iniparam['q_ven'+str(self.vs+1)+'_sys_0']
        except: var[self.varmap['q_ven'+str(self.vs+1)+'_sys']]    = iniparam['q_ven_sys_0']


    def print_to_screen(self, var_sq, aux):

        print("Output of 0D coronary model (ZCRp_CRd_lr):")

        print('{:<12s}{:<3s}{:<7.3f}{:<3s}{:<12s}{:<3s}{:<7.3f}'.format('p_cord_sys_l',' = ',var_sq[self.varmap['p_cord_sys_l']],'   ','p_cord_sys_r',' = ',var_sq[self.varmap['p_cord_sys_r']]))
        sys.stdout.flush()
