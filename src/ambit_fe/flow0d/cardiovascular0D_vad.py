#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import sympy as sp


class vad_circ():

    def __init__(self, params, varmap, auxmap):

        # tab. 1, "8k"
        self.A = 1.17e1 # kPa
        self.B = -7.72e-5 # kPa s/mm^3
        self.C = 0. # kPa s^2/mm^6

        self.ndvad = 2

        self.varmap = varmap
        self.auxmap = auxmap


    def equation_map(self, vindex, aindex, x_, a_, df_, f_, p_v_, p_ar_):

        self.varmap['q_vad_in']  = vindex
        self.varmap['q_vad_out'] = vindex+1

        q_vad_in_  = sp.Symbol('q_vad_in_')
        q_vad_out_ = sp.Symbol('q_vad_out_')

        x_[self.varmap['q_vad_in']]  = q_vad_in_
        x_[self.varmap['q_vad_out']] = q_vad_out_

        # simple VAD model according to Santiago et al. 2022, PLoS Comput Biol 18(6): e1010141, doi: 10.1371/journal.pcbi.1010141
        # p_ao - p_lv = A + B * Q_vad + C * Q_vad^2

        # populate df_ and f_ arrays
        df_[vindex]   = 0.
        df_[vindex+1] = 0.

        f_[vindex]   = q_vad_out_ - q_vad_in_
        f_[vindex+1] = (p_ar_ - p_v_ - self.A)/self.B - q_vad_out_ - (self.C/self.B) * q_vad_out_**2.

        return q_vad_in_, q_vad_out_


    def initialize(self, var, iniparam):

        ### Implement your initialize routine here
        pass



    def print_to_screen(self, var_sq, aux):

        print("Output of 0D VAD model (type):")
        ### Implement your printout routine here
