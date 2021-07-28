#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sympy as sp

class coronary_circ():
    
    def __init__(self, params):
        
        self.R_arcor_sys = params['R_arcor_sys']
        self.C_arcor_sys = params['C_arcor_sys']
    
    
    # simplest form of coronary circulation model: RC model (2-element Windkessel)
    def equation_map(self, mapindex, x_, df_, f_, p_ar_, p_corsinus_, q_in_, q_out_):
        
        # no need to populate x_ for this model (parent circulation holds in- and outflows, artery and coronary sinus terminal pressure)

        # populate df_ and f_ arrays
        df_[mapindex]   = self.C_arcor_sys * p_ar_                                     # coronary volume rate
        df_[mapindex+1] = 0.
        
        f_[mapindex]    = q_out_ - q_in_                                               # coronary flow balance
        f_[mapindex+1]  = (p_corsinus_ - p_ar_)/self.R_arcor_sys + q_out_              # coronary momentum
