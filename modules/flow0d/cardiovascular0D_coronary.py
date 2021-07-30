#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import sympy as sp

class coronary_circ_RCsimple():
    
    def __init__(self, params, varmap, auxmap, vs):
        
        self.R_arcor_sys = params['R_arcor_sys']
        self.C_arcor_sys = params['C_arcor_sys']
        
        self.R_vencor_sys = params['R_vencor_sys']
        self.C_vencor_sys = params['C_vencor_sys']
        
        self.V_arcor_sys_u = params['V_arcor_sys_u']
        self.V_vencor_sys_u = params['V_vencor_sys_u']
        
        self.ndcor = 4
        
        self.varmap = varmap
        self.auxmap = auxmap

        self.vs = vs
    
    # simplest form of coronary circulation model: RC model (2-element Windkessel)
    def equation_map(self, vindex, aindex, x_, a_, df_, f_, p_ar_, p_at):
        
        self.varmap['q_arcor_sys_in']         = vindex
        self.varmap['q_arcor_sys']            = vindex+1
        self.varmap['p_vencor_sys']           = vindex+2
        self.varmap['q_ven'+str(self.vs+1)+'_sys'] = vindex+3

        q_arcor_sys_in_   = sp.Symbol('q_arcor_sys_in_')
        q_arcor_sys_      = sp.Symbol('q_arcor_sys_')
        p_vencor_sys_     = sp.Symbol('p_vencor_sys_')
        q_vencor_sys_out_ = sp.Symbol('q_ven'+str(self.vs+1)+'_sys_')
        
        x_[self.varmap['q_arcor_sys_in']]         = q_arcor_sys_in_
        x_[self.varmap['q_arcor_sys']]            = q_arcor_sys_
        x_[self.varmap['p_vencor_sys']]           = p_vencor_sys_
        x_[self.varmap['q_ven'+str(self.vs+1)+'_sys']] = q_vencor_sys_out_

        # populate df_ and f_ arrays
        df_[vindex]   = self.C_arcor_sys * p_ar_                                    # coronary arterial volume rate
        df_[vindex+1] = 0.
        df_[vindex+2] = self.C_vencor_sys * p_vencor_sys_                           # coronary venous/coronary sinus volume rate
        df_[vindex+3] = 0
        
        f_[vindex]   = q_arcor_sys_ - q_arcor_sys_in_                               # coronary arterial flow balance
        f_[vindex+1] = (p_vencor_sys_ - p_ar_)/self.R_arcor_sys + q_arcor_sys_      # coronary arterial momentum
        f_[vindex+2] = q_vencor_sys_out_ - q_arcor_sys_                             # coronary venous/coronary sinus flow balance
        f_[vindex+3] = (p_at - p_vencor_sys_)/self.R_vencor_sys + q_vencor_sys_out_ # coronary venous/coronary sinus momentum

        
        # auxiliary map and variables
        self.auxmap['V_arcor_sys']  = aindex
        self.auxmap['V_vencor_sys'] = aindex+1
        
        a_[self.auxmap['V_arcor_sys']]  = self.C_arcor_sys * p_ar_ + self.V_arcor_sys_u
        a_[self.auxmap['V_vencor_sys']] = self.C_vencor_sys * p_vencor_sys_ + self.V_vencor_sys_u
        
        return q_arcor_sys_in_, q_vencor_sys_out_


    def initialize(self, var, iniparam):
        
        try: var[self.varmap['q_arcor_sys_in']]                 = iniparam['q_arcor_sys_in_0']
        except: var[self.varmap['q_arcor_sys_in']]                 = iniparam['q_arcor_sys_0']
        var[self.varmap['q_arcor_sys']]                         = iniparam['q_arcor_sys_0']
        
        var[self.varmap['p_vencor_sys']]                        = iniparam['p_vencor_sys_0']
        
        try: var[self.varmap['q_ven'+str(self.vs+1)+'_sys']]    = iniparam['q_ven'+str(self.vs+1)+'_sys_0']
        except: var[self.varmap['q_ven'+str(self.vs+1)+'_sys']]    = iniparam['q_ven_sys_0']
