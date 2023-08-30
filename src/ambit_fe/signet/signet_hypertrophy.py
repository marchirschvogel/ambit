#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sympy as sp

from ..mpiroutines import allgather_vec
from ..oderoutines import ode

# signalling network model, from Ryall et al. (2012) "Network Reconstruction and Systems Analysis of Cardiac Myocyte Hypertrophy Signaling", The Journal of Biological Chemistry 287(50)
# adopted form supplementary MATLAB code provided by authors

class signethypertrophy(ode):

    def __init__(self, params, init=True, comm=None):

        # initialize base class
        super().__init__(init=init, comm=comm)

        # parameters
        self.p1 = params['p1']

        self.params = params

        # set up arrays
        self.setup_arrays()

        # set up symbolic equations
        self.equation_map()

        # symbolic stiffness matrix
        self.set_stiffness()

        # make Lambda functions out of symbolic expressions
        self.lambdify_expressions()


    def setup_arrays(self):

        # number of degrees of freedom
        self.numdof = 106

        self.set_solve_arrays()


    def equation_map(self):

        # variable map
        self.varmap['aAR']       = 0
        self.varmap['AC']        = 1
        self.varmap['Akt']       = 2
        self.varmap['aMHC']      = 3
        self.varmap['AngII']     = 4
        self.varmap['ANP']       = 5
        self.varmap['ANPi']      = 6
        self.varmap['AT1R']      = 7
        self.varmap['ATF2']      = 8
        self.varmap['BAR']       = 9
        self.varmap['bMHC']      = 10
        self.varmap['BNP']       = 11
        self.varmap['BNPi']      = 12
        self.varmap['Calcium']   = 13
        self.varmap['CaM']       = 14
        self.varmap['CaMK']      = 15
        self.varmap['cAMP']      = 16
        self.varmap['CaN']       = 17
        self.varmap['CellArea']  = 18
        self.varmap['cFos']      = 19
        self.varmap['cGMP']      = 20
        self.varmap['cJun']      = 21
        self.varmap['CREB']      = 22
        self.varmap['CT1']       = 23
        self.varmap['DAG']       = 24
        self.varmap['EGF']       = 25
        self.varmap['EGFR']      = 26
        self.varmap['eIF2B']     = 27
        self.varmap['eIF4E']     = 28
        self.varmap['ELK1']      = 29
        self.varmap['ERBB']      = 30
        self.varmap['ERK12']     = 31
        self.varmap['ERK5']      = 32
        self.varmap['ET1']       = 33
        self.varmap['ET1R']      = 34
        self.varmap['FAK']       = 35
        self.varmap['FGF']       = 36
        self.varmap['FGFR']      = 37
        self.varmap['foxo']      = 38
        self.varmap['Gaq11']     = 39
        self.varmap['GATA4']     = 40
        self.varmap['GBG']       = 41
        self.varmap['GCA']       = 42
        self.varmap['gp130LIFR'] = 43
        self.varmap['Gsa']       = 44
        self.varmap['GSK3B']     = 45
        self.varmap['HDAC']      = 46
        self.varmap['IGF1']      = 47
        self.varmap['IGF1R']     = 48
        self.varmap['IkB']       = 49
        self.varmap['IKK']       = 50
        self.varmap['IL6']       = 51
        self.varmap['IL6R']      = 52
        self.varmap['Integrins'] = 53
        self.varmap['IP3']       = 54
        self.varmap['ISO']       = 55
        self.varmap['JAK']       = 56
        self.varmap['JNK']       = 57
        self.varmap['LIF']       = 58
        self.varmap['MAP3K11']   = 59
        self.varmap['MAP3K23']   = 60
        self.varmap['MAP3K4']    = 61
        self.varmap['MAPKAPK']   = 62
        self.varmap['MEF2']      = 63
        self.varmap['MEK12']     = 64
        self.varmap['MEK36']     = 65
        self.varmap['MEK4']      = 66
        self.varmap['MEK5']      = 67
        self.varmap['MEK7']      = 68
        self.varmap['MEKK1']     = 69
        self.varmap['MSK1']      = 70
        self.varmap['mTor']      = 71
        self.varmap['NE']        = 72
        self.varmap['NFAT']      = 73
        self.varmap['NFkB']      = 74
        self.varmap['NIK']       = 75
        self.varmap['NOS']       = 76
        self.varmap['NRG1']      = 77
        self.varmap['p38']       = 78
        self.varmap['p70s6k']    = 79
        self.varmap['PDK1']      = 80
        self.varmap['PE']        = 81
        self.varmap['PI3K']      = 82
        self.varmap['PKA']       = 83
        self.varmap['PKC']       = 84
        self.varmap['PKD']       = 85
        self.varmap['PKG1']      = 86
        self.varmap['PLCB']      = 87
        self.varmap['PLCG']      = 88
        self.varmap['Rac1']      = 89
        self.varmap['Raf1']      = 90
        self.varmap['Raf1A']     = 91
        self.varmap['Ras']       = 92
        self.varmap['RhoA']      = 93
        self.varmap['sACT']      = 94
        self.varmap['SERCA']     = 95
        self.varmap['sGC']       = 96
        self.varmap['SHP2']      = 97
        self.varmap['SRF']       = 98
        self.varmap['STAT']      = 99
        self.varmap['Stretch']   = 100
        self.varmap['TAK1']      = 101
        self.varmap['TGFB']      = 102
        self.varmap['TGFR']      = 103
        self.varmap['TNFa']      = 104
        self.varmap['TNFR']      = 105

        self.t_ = sp.Symbol('t_')

        for k in self.varmap.keys():
            self.x_[self.varmap[k]] = sp.Symbol(k+'_')

        tau, xmax = np.ones(self.numdof), np.ones(self.numdof)

        # reaction parameters
        rpar = np.zeros((3,193))

        # w
        rpar[0,:] = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
        # n
        rpar[1,:] = [1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000, 1.400000e+000]
        # EC50
        rpar[2,:] = [5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001, 5.000000e-001]

        for k in self.varmap.keys():
            self.df_[self.varmap[k]] = self.x_[self.varmap[k]]

        self.f_[self.varmap['aAR']]       = -(self.OR(self.act(self.x_[self.varmap['NE']],rpar[:,143]),self.act(self.x_[self.varmap['PE']],rpar[:,160]))*xmax[self.varmap['aAR']] - self.x_[self.varmap['aAR']])/tau[self.varmap['aAR']]
        self.f_[self.varmap['AC']]        = -(self.act(self.x_[self.varmap['Gsa']],rpar[:,99])*xmax[self.varmap['AC']] - self.x_[self.varmap['AC']])/tau[self.varmap['AC']]
        self.f_[self.varmap['Akt']]       = -(self.act(self.x_[self.varmap['PDK1']],rpar[:,159])*xmax[self.varmap['Akt']] - self.x_[self.varmap['Akt']])/tau[self.varmap['Akt']]
        self.f_[self.varmap['aMHC']]      = -(self.inhib(self.x_[self.varmap['cFos']],rpar[:,20])*self.inhib(self.x_[self.varmap['cJun']],rpar[:,20])*self.inhib(self.x_[self.varmap['MEF2']],rpar[:,20])*self.inhib(self.x_[self.varmap['NFAT']],rpar[:,20])*xmax[self.varmap['aMHC']] - self.x_[self.varmap['aMHC']])/tau[self.varmap['aMHC']]
        self.f_[self.varmap['AngII']]     = -(rpar[0,0]*xmax[self.varmap['AngII']] - self.x_[self.varmap['AngII']])/tau[self.varmap['AngII']]
        self.f_[self.varmap['ANP']]       = -(self.OR(self.act(self.x_[self.varmap['ATF2']],rpar[:,42]),self.OR(self.act(self.x_[self.varmap['cFos']],rpar[:,54]),self.OR(self.act(self.x_[self.varmap['cJun']],rpar[:,59]),self.OR(self.act(self.x_[self.varmap['CREB']],rpar[:,63]),self.OR(self.act(self.x_[self.varmap['MEF2']],rpar[:,126]),self.OR(self.act(self.x_[self.varmap['GATA4']],rpar[:,145])*self.act(self.x_[self.varmap['NFAT']],rpar[:,145]),self.act(self.x_[self.varmap['STAT']],rpar[:,184])))))))*xmax[self.varmap['ANP']] - self.x_[self.varmap['ANP']])/tau[self.varmap['ANP']]
        self.f_[self.varmap['ANPi']]      = -(rpar[0,1]*xmax[self.varmap['ANPi']] - self.x_[self.varmap['ANPi']])/tau[self.varmap['ANPi']]
        self.f_[self.varmap['AT1R']]      = -(self.act(self.x_[self.varmap['AngII']],rpar[:,38])*xmax[self.varmap['AT1R']] - self.x_[self.varmap['AT1R']])/tau[self.varmap['AT1R']]
        self.f_[self.varmap['ATF2']]      = -(self.OR(self.act(self.x_[self.varmap['JNK']],rpar[:,113]),self.act(self.x_[self.varmap['p38']],rpar[:,152]))*xmax[self.varmap['ATF2']] - self.x_[self.varmap['ATF2']])/tau[self.varmap['ATF2']]
        self.f_[self.varmap['BAR']]       = -(self.OR(self.act(self.x_[self.varmap['ISO']],rpar[:,109]),self.act(self.x_[self.varmap['NE']],rpar[:,144]))*xmax[self.varmap['BAR']] - self.x_[self.varmap['BAR']])/tau[self.varmap['BAR']]
        self.f_[self.varmap['bMHC']]      = -(self.OR(self.act(self.x_[self.varmap['ATF2']],rpar[:,43]),self.OR(self.act(self.x_[self.varmap['cFos']],rpar[:,55]),self.OR(self.act(self.x_[self.varmap['cJun']],rpar[:,60]),self.OR(self.act(self.x_[self.varmap['GATA4']],rpar[:,91]),self.OR(self.act(self.x_[self.varmap['MEF2']],rpar[:,127]),self.OR(self.act(self.x_[self.varmap['NFAT']],rpar[:,147]),self.act(self.x_[self.varmap['STAT']],rpar[:,185])))))))*xmax[self.varmap['bMHC']] - self.x_[self.varmap['bMHC']])/tau[self.varmap['bMHC']]
        self.f_[self.varmap['BNP']]       = -(self.OR(self.act(self.x_[self.varmap['ATF2']],rpar[:,44]),self.OR(self.act(self.x_[self.varmap['cFos']],rpar[:,56]),self.OR(self.act(self.x_[self.varmap['cJun']],rpar[:,61]),self.OR(self.act(self.x_[self.varmap['ELK1']],rpar[:,71]),self.OR(self.act(self.x_[self.varmap['MEF2']],rpar[:,128]),self.act(self.x_[self.varmap['GATA4']],rpar[:,146])*self.act(self.x_[self.varmap['NFAT']],rpar[:,146]))))))*xmax[self.varmap['BNP']] - self.x_[self.varmap['BNP']])/tau[self.varmap['BNP']]
        self.f_[self.varmap['BNPi']]      = -(rpar[0,2]*xmax[self.varmap['BNPi']] - self.x_[self.varmap['BNPi']])/tau[self.varmap['BNPi']]
        self.f_[self.varmap['Calcium']]   = -(self.OR(self.act(self.x_[self.varmap['IP3']],rpar[:,108]),self.act(self.x_[self.varmap['PKA']],rpar[:,162]))*xmax[self.varmap['Calcium']] - self.x_[self.varmap['Calcium']])/tau[self.varmap['Calcium']]
        self.f_[self.varmap['CaM']]       = -(self.act(self.x_[self.varmap['Calcium']],rpar[:,48])*xmax[self.varmap['CaM']] - self.x_[self.varmap['CaM']])/tau[self.varmap['CaM']]
        self.f_[self.varmap['CaMK']]      = -(self.act(self.x_[self.varmap['CaM']],rpar[:,49])*xmax[self.varmap['CaMK']] - self.x_[self.varmap['CaMK']])/tau[self.varmap['CaMK']]
        self.f_[self.varmap['cAMP']]      = -(self.act(self.x_[self.varmap['AC']],rpar[:,34])*xmax[self.varmap['cAMP']] - self.x_[self.varmap['cAMP']])/tau[self.varmap['cAMP']]
        self.f_[self.varmap['CaN']]       = -(self.act(self.x_[self.varmap['CaM']],rpar[:,50])*xmax[self.varmap['CaN']] - self.x_[self.varmap['CaN']])/tau[self.varmap['CaN']]
        self.f_[self.varmap['CellArea']]  = -(self.OR(self.inhib(self.x_[self.varmap['foxo']],rpar[:,22]),self.OR(self.act(self.x_[self.varmap['ATF2']],rpar[:,45]),self.OR(self.act(self.x_[self.varmap['cJun']],rpar[:,62]),self.OR(self.act(self.x_[self.varmap['CREB']],rpar[:,64]),self.OR(self.act(self.x_[self.varmap['GATA4']],rpar[:,92]),self.act(self.x_[self.varmap['MEF2']],rpar[:,129]))))))*xmax[self.varmap['CellArea']] - self.x_[self.varmap['CellArea']])/tau[self.varmap['CellArea']]
        self.f_[self.varmap['cFos']]      = -(self.act(self.x_[self.varmap['ERK12']],rpar[:,76])*xmax[self.varmap['cFos']] - self.x_[self.varmap['cFos']])/tau[self.varmap['cFos']]
        self.f_[self.varmap['cGMP']]      = -(self.OR(self.act(self.x_[self.varmap['GCA']],rpar[:,96]),self.act(self.x_[self.varmap['sGC']],rpar[:,182]))*xmax[self.varmap['cGMP']] - self.x_[self.varmap['cGMP']])/tau[self.varmap['cGMP']]
        self.f_[self.varmap['cJun']]      = -(self.OR(self.act(self.x_[self.varmap['ERK12']],rpar[:,77]),self.act(self.x_[self.varmap['JNK']],rpar[:,114]))*xmax[self.varmap['cJun']] - self.x_[self.varmap['cJun']])/tau[self.varmap['cJun']]
        self.f_[self.varmap['CREB']]      = -(self.OR(self.inhib(self.x_[self.varmap['GSK3B']],rpar[:,24]),self.OR(self.act(self.x_[self.varmap['MAPKAPK']],rpar[:,125]),self.OR(self.act(self.x_[self.varmap['MSK1']],rpar[:,140]),self.act(self.x_[self.varmap['PKA']],rpar[:,163]))))*xmax[self.varmap['CREB']] - self.x_[self.varmap['CREB']])/tau[self.varmap['CREB']]
        self.f_[self.varmap['CT1']]       = -(rpar[0,3]*xmax[self.varmap['CT1']] - self.x_[self.varmap['CT1']])/tau[self.varmap['CT1']]
        self.f_[self.varmap['DAG']]       = -(self.OR(self.act(self.x_[self.varmap['PLCB']],rpar[:,168]),self.act(self.x_[self.varmap['PLCG']],rpar[:,170]))*xmax[self.varmap['DAG']] - self.x_[self.varmap['DAG']])/tau[self.varmap['DAG']]
        self.f_[self.varmap['EGF']]       = -(rpar[0,4]*xmax[self.varmap['EGF']] - self.x_[self.varmap['EGF']])/tau[self.varmap['EGF']]
        self.f_[self.varmap['EGFR']]      = -(self.act(self.x_[self.varmap['EGF']],rpar[:,67])*xmax[self.varmap['EGFR']] - self.x_[self.varmap['EGFR']])/tau[self.varmap['EGFR']]
        self.f_[self.varmap['eIF2B']]     = -(self.inhib(self.x_[self.varmap['GSK3B']],rpar[:,25])*xmax[self.varmap['eIF2B']] - self.x_[self.varmap['eIF2B']])/tau[self.varmap['eIF2B']]
        self.f_[self.varmap['eIF4E']]     = -(self.act(self.x_[self.varmap['mTor']],rpar[:,141])*xmax[self.varmap['eIF4E']] - self.x_[self.varmap['eIF4E']])/tau[self.varmap['eIF4E']]
        self.f_[self.varmap['ELK1']]      = -(self.OR(self.act(self.x_[self.varmap['ERK12']],rpar[:,78]),self.OR(self.act(self.x_[self.varmap['JNK']],rpar[:,115]),self.act(self.x_[self.varmap['p38']],rpar[:,153])))*xmax[self.varmap['ELK1']] - self.x_[self.varmap['ELK1']])/tau[self.varmap['ELK1']]
        self.f_[self.varmap['ERBB']]      = -(self.act(self.x_[self.varmap['NRG1']],rpar[:,151])*xmax[self.varmap['ERBB']] - self.x_[self.varmap['ERBB']])/tau[self.varmap['ERBB']]
        self.f_[self.varmap['ERK12']]     = -(self.act(self.x_[self.varmap['MEK12']],rpar[:,131])*xmax[self.varmap['ERK12']] - self.x_[self.varmap['ERK12']])/tau[self.varmap['ERK12']]
        self.f_[self.varmap['ERK5']]      = -(self.act(self.x_[self.varmap['MEK5']],rpar[:,135])*xmax[self.varmap['ERK5']] - self.x_[self.varmap['ERK5']])/tau[self.varmap['ERK5']]
        self.f_[self.varmap['ET1']]       = -(rpar[0,5]*xmax[self.varmap['ET1']] - self.x_[self.varmap['ET1']])/tau[self.varmap['ET1']]
        self.f_[self.varmap['ET1R']]      = -(self.act(self.x_[self.varmap['ET1']],rpar[:,83])*xmax[self.varmap['ET1R']] - self.x_[self.varmap['ET1R']])/tau[self.varmap['ET1R']]
        self.f_[self.varmap['FAK']]       = -(self.act(self.x_[self.varmap['Integrins']],rpar[:,107])*xmax[self.varmap['FAK']] - self.x_[self.varmap['FAK']])/tau[self.varmap['FAK']]
        self.f_[self.varmap['FGF']]       = -(rpar[0,6]*xmax[self.varmap['FGF']] - self.x_[self.varmap['FGF']])/tau[self.varmap['FGF']]
        self.f_[self.varmap['FGFR']]      = -(self.act(self.x_[self.varmap['FGF']],rpar[:,86])*xmax[self.varmap['FGFR']] - self.x_[self.varmap['FGFR']])/tau[self.varmap['FGFR']]
        self.f_[self.varmap['foxo']]      = -(self.inhib(self.x_[self.varmap['Akt']],rpar[:,17])*xmax[self.varmap['foxo']] - self.x_[self.varmap['foxo']])/tau[self.varmap['foxo']]
        self.f_[self.varmap['Gaq11']]     = -(self.OR(self.act(self.x_[self.varmap['aAR']],rpar[:,33]),self.OR(self.act(self.x_[self.varmap['AT1R']],rpar[:,40]),self.act(self.x_[self.varmap['ET1R']],rpar[:,84])))*xmax[self.varmap['Gaq11']] - self.x_[self.varmap['Gaq11']])/tau[self.varmap['Gaq11']]
        self.f_[self.varmap['GATA4']]     = -(self.OR(self.inhib(self.x_[self.varmap['GSK3B']],rpar[:,26]),self.OR(self.act(self.x_[self.varmap['ERK12']],rpar[:,79]),self.act(self.x_[self.varmap['p38']],rpar[:,154])))*xmax[self.varmap['GATA4']] - self.x_[self.varmap['GATA4']])/tau[self.varmap['GATA4']]
        self.f_[self.varmap['GBG']]       = -(self.OR(self.act(self.x_[self.varmap['Gaq11']],rpar[:,88]),self.act(self.x_[self.varmap['Gsa']],rpar[:,100]))*xmax[self.varmap['GBG']] - self.x_[self.varmap['GBG']])/tau[self.varmap['GBG']]
        self.f_[self.varmap['GCA']]       = -(self.OR(self.act(self.x_[self.varmap['ANPi']],rpar[:,39]),self.act(self.x_[self.varmap['BNPi']],rpar[:,47]))*xmax[self.varmap['GCA']] - self.x_[self.varmap['GCA']])/tau[self.varmap['GCA']]
        self.f_[self.varmap['gp130LIFR']] = -(self.OR(self.act(self.x_[self.varmap['CT1']],rpar[:,65]),self.act(self.x_[self.varmap['LIF']],rpar[:,116]))*xmax[self.varmap['gp130LIFR']] - self.x_[self.varmap['gp130LIFR']])/tau[self.varmap['gp130LIFR']]
        self.f_[self.varmap['Gsa']]       = -(self.act(self.x_[self.varmap['BAR']],rpar[:,46])*xmax[self.varmap['Gsa']] - self.x_[self.varmap['Gsa']])/tau[self.varmap['Gsa']]
        self.f_[self.varmap['GSK3B']]     = -(self.inhib(self.x_[self.varmap['Akt']],rpar[:,18])*xmax[self.varmap['GSK3B']] - self.x_[self.varmap['GSK3B']])/tau[self.varmap['GSK3B']]
        self.f_[self.varmap['HDAC']]      = -(self.OR(self.inhib(self.x_[self.varmap['CaMK']],rpar[:,19]),self.OR(self.inhib(self.x_[self.varmap['PKC']],rpar[:,31]),self.inhib(self.x_[self.varmap['PKD']],rpar[:,32])))*xmax[self.varmap['HDAC']] - self.x_[self.varmap['HDAC']])/tau[self.varmap['HDAC']]
        self.f_[self.varmap['IGF1']]      = -(rpar[0,7]*xmax[self.varmap['IGF1']] - self.x_[self.varmap['IGF1']])/tau[self.varmap['IGF1']]
        self.f_[self.varmap['IGF1R']]     = -(self.act(self.x_[self.varmap['IGF1']],rpar[:,101])*xmax[self.varmap['IGF1R']] - self.x_[self.varmap['IGF1R']])/tau[self.varmap['IGF1R']]
        self.f_[self.varmap['IkB']]       = -(self.inhib(self.x_[self.varmap['IKK']],rpar[:,29])*xmax[self.varmap['IkB']] - self.x_[self.varmap['IkB']])/tau[self.varmap['IkB']]
        self.f_[self.varmap['IKK']]       = -(self.OR(self.act(self.x_[self.varmap['Akt']],rpar[:,35]),self.OR(self.act(self.x_[self.varmap['NIK']],rpar[:,149]),self.act(self.x_[self.varmap['p38']],rpar[:,155])))*xmax[self.varmap['IKK']] - self.x_[self.varmap['IKK']])/tau[self.varmap['IKK']]
        self.f_[self.varmap['IL6']]       = -(rpar[0,8]*xmax[self.varmap['IL6']] - self.x_[self.varmap['IL6']])/tau[self.varmap['IL6']]
        self.f_[self.varmap['IL6R']]      = -(self.act(self.x_[self.varmap['IL6']],rpar[:,105])*xmax[self.varmap['IL6R']] - self.x_[self.varmap['IL6R']])/tau[self.varmap['IL6R']]
        self.f_[self.varmap['Integrins']] = -(self.act(self.x_[self.varmap['Stretch']],rpar[:,186])*xmax[self.varmap['Integrins']] - self.x_[self.varmap['Integrins']])/tau[self.varmap['Integrins']]
        self.f_[self.varmap['IP3']]       = -(self.OR(self.act(self.x_[self.varmap['PLCB']],rpar[:,169]),self.act(self.x_[self.varmap['PLCG']],rpar[:,171]))*xmax[self.varmap['IP3']] - self.x_[self.varmap['IP3']])/tau[self.varmap['IP3']]
        self.f_[self.varmap['ISO']]       = -(rpar[0,9]*xmax[self.varmap['ISO']] - self.x_[self.varmap['ISO']])/tau[self.varmap['ISO']]
        self.f_[self.varmap['JAK']]       = -(self.OR(self.act(self.x_[self.varmap['AT1R']],rpar[:,41]),self.OR(self.act(self.x_[self.varmap['gp130LIFR']],rpar[:,97]),self.act(self.x_[self.varmap['IL6R']],rpar[:,106])))*xmax[self.varmap['JAK']] - self.x_[self.varmap['JAK']])/tau[self.varmap['JAK']]
        self.f_[self.varmap['JNK']]       = -(self.OR(self.act(self.x_[self.varmap['MEK4']],rpar[:,133]),self.act(self.x_[self.varmap['MEK7']],rpar[:,136]))*xmax[self.varmap['JNK']] - self.x_[self.varmap['JNK']])/tau[self.varmap['JNK']]
        self.f_[self.varmap['LIF']]       = -(rpar[0,10]*xmax[self.varmap['LIF']] - self.x_[self.varmap['LIF']])/tau[self.varmap['LIF']]
        self.f_[self.varmap['MAP3K11']]   = -(self.act(self.x_[self.varmap['Rac1']],rpar[:,172])*xmax[self.varmap['MAP3K11']] - self.x_[self.varmap['MAP3K11']])/tau[self.varmap['MAP3K11']]
        self.f_[self.varmap['MAP3K23']]   = -(self.act(self.x_[self.varmap['Ras']],rpar[:,176])*xmax[self.varmap['MAP3K23']] - self.x_[self.varmap['MAP3K23']])/tau[self.varmap['MAP3K23']]
        self.f_[self.varmap['MAP3K4']]    = -(self.act(self.x_[self.varmap['Rac1']],rpar[:,173])*xmax[self.varmap['MAP3K4']] - self.x_[self.varmap['MAP3K4']])/tau[self.varmap['MAP3K4']]
        self.f_[self.varmap['MAPKAPK']]   = -(self.act(self.x_[self.varmap['p38']],rpar[:,156])*xmax[self.varmap['MAPKAPK']] - self.x_[self.varmap['MAPKAPK']])/tau[self.varmap['MAPKAPK']]
        self.f_[self.varmap['MEF2']]      = -(self.OR(self.inhib(self.x_[self.varmap['HDAC']],rpar[:,27]),self.OR(self.act(self.x_[self.varmap['ERK5']],rpar[:,82]),self.act(self.x_[self.varmap['p38']],rpar[:,157])))*xmax[self.varmap['MEF2']] - self.x_[self.varmap['MEF2']])/tau[self.varmap['MEF2']]
        self.f_[self.varmap['MEK12']]     = -(self.OR(self.act(self.x_[self.varmap['MAP3K23']],rpar[:,119]),self.OR(self.act(self.x_[self.varmap['MEKK1']],rpar[:,137]),self.act(self.x_[self.varmap['Raf1']],rpar[:,174])))*xmax[self.varmap['MEK12']] - self.x_[self.varmap['MEK12']])/tau[self.varmap['MEK12']]
        self.f_[self.varmap['MEK36']]     = -(self.OR(self.act(self.x_[self.varmap['MAP3K11']],rpar[:,117]),self.act(self.x_[self.varmap['TAK1']],rpar[:,187]))*xmax[self.varmap['MEK36']] - self.x_[self.varmap['MEK36']])/tau[self.varmap['MEK36']]
        self.f_[self.varmap['MEK4']]      = -(self.OR(self.act(self.x_[self.varmap['MAP3K23']],rpar[:,120]),self.OR(self.act(self.x_[self.varmap['MAP3K4']],rpar[:,123]),self.act(self.x_[self.varmap['MEKK1']],rpar[:,138])))*xmax[self.varmap['MEK4']] - self.x_[self.varmap['MEK4']])/tau[self.varmap['MEK4']]
        self.f_[self.varmap['MEK5']]      = -(self.OR(self.act(self.x_[self.varmap['MAP3K23']],rpar[:,121]),self.act(self.x_[self.varmap['SHP2']],rpar[:,183]))*xmax[self.varmap['MEK5']] - self.x_[self.varmap['MEK5']])/tau[self.varmap['MEK5']]
        self.f_[self.varmap['MEK7']]      = -(self.OR(self.act(self.x_[self.varmap['MAP3K11']],rpar[:,118]),self.OR(self.act(self.x_[self.varmap['MAP3K23']],rpar[:,122]),self.OR(self.act(self.x_[self.varmap['MAP3K4']],rpar[:,124]),self.act(self.x_[self.varmap['MEKK1']],rpar[:,139]))))*xmax[self.varmap['MEK7']] - self.x_[self.varmap['MEK7']])/tau[self.varmap['MEK7']]
        self.f_[self.varmap['MEKK1']]     = -(self.act(self.x_[self.varmap['Ras']],rpar[:,177])*xmax[self.varmap['MEKK1']] - self.x_[self.varmap['MEKK1']])/tau[self.varmap['MEKK1']]
        self.f_[self.varmap['MSK1']]      = -(self.OR(self.act(self.x_[self.varmap['ERK12']],rpar[:,80]),self.act(self.x_[self.varmap['p38']],rpar[:,158]))*xmax[self.varmap['MSK1']] - self.x_[self.varmap['MSK1']])/tau[self.varmap['MSK1']]
        self.f_[self.varmap['mTor']]      = -(self.act(self.x_[self.varmap['Akt']],rpar[:,36])*xmax[self.varmap['mTor']] - self.x_[self.varmap['mTor']])/tau[self.varmap['mTor']]
        self.f_[self.varmap['NE']]        = -(rpar[0,11]*xmax[self.varmap['NE']] - self.x_[self.varmap['NE']])/tau[self.varmap['NE']]
        self.f_[self.varmap['NFAT']]      = -(self.OR(self.inhib(self.x_[self.varmap['GSK3B']],rpar[:,23])*self.inhib(self.x_[self.varmap['JNK']],rpar[:,23])*self.inhib(self.x_[self.varmap['p38']],rpar[:,23])*self.inhib(self.x_[self.varmap['PKA']],rpar[:,23])*self.inhib(self.x_[self.varmap['PKG1']],rpar[:,23]),self.OR(self.act(self.x_[self.varmap['CaN']],rpar[:,52]),self.act(self.x_[self.varmap['CaN']],rpar[:,75])*self.act(self.x_[self.varmap['ERK12']],rpar[:,75])))*xmax[self.varmap['NFAT']] - self.x_[self.varmap['NFAT']])/tau[self.varmap['NFAT']]
        self.f_[self.varmap['NFkB']]      = -(self.OR(self.inhib(self.x_[self.varmap['IkB']],rpar[:,28]),self.act(self.x_[self.varmap['ERK12']],rpar[:,81]))*xmax[self.varmap['NFkB']] - self.x_[self.varmap['NFkB']])/tau[self.varmap['NFkB']]
        self.f_[self.varmap['NIK']]       = -(self.act(self.x_[self.varmap['TNFR']],rpar[:,191])*xmax[self.varmap['NIK']] - self.x_[self.varmap['NIK']])/tau[self.varmap['NIK']]
        self.f_[self.varmap['NOS']]       = -(self.act(self.x_[self.varmap['Akt']],rpar[:,37])*xmax[self.varmap['NOS']] - self.x_[self.varmap['NOS']])/tau[self.varmap['NOS']]
        self.f_[self.varmap['NRG1']]      = -(rpar[0,12]*xmax[self.varmap['NRG1']] - self.x_[self.varmap['NRG1']])/tau[self.varmap['NRG1']]
        self.f_[self.varmap['p38']]       = -(self.OR(self.act(self.x_[self.varmap['MEK36']],rpar[:,132]),self.act(self.x_[self.varmap['MEK4']],rpar[:,134]))*xmax[self.varmap['p38']] - self.x_[self.varmap['p38']])/tau[self.varmap['p38']]
        self.f_[self.varmap['p70s6k']]    = -(self.act(self.x_[self.varmap['mTor']],rpar[:,142])*xmax[self.varmap['p70s6k']] - self.x_[self.varmap['p70s6k']])/tau[self.varmap['p70s6k']]
        self.f_[self.varmap['PDK1']]      = -(self.act(self.x_[self.varmap['PI3K']],rpar[:,161])*xmax[self.varmap['PDK1']] - self.x_[self.varmap['PDK1']])/tau[self.varmap['PDK1']]
        self.f_[self.varmap['PE']]        = -(rpar[0,13]*xmax[self.varmap['PE']] - self.x_[self.varmap['PE']])/tau[self.varmap['PE']]
        self.f_[self.varmap['PI3K']]      = -(self.OR(self.act(self.x_[self.varmap['EGFR']],rpar[:,68]),self.OR(self.act(self.x_[self.varmap['ERBB']],rpar[:,72]),self.OR(self.act(self.x_[self.varmap['GBG']],rpar[:,93]),self.OR(self.act(self.x_[self.varmap['IGF1R']],rpar[:,102]),self.OR(self.act(self.x_[self.varmap['JAK']],rpar[:,110]),self.OR(self.act(self.x_[self.varmap['Ras']],rpar[:,178]),self.act(self.x_[self.varmap['TNFR']],rpar[:,192])))))))*xmax[self.varmap['PI3K']] - self.x_[self.varmap['PI3K']])/tau[self.varmap['PI3K']]
        self.f_[self.varmap['PKA']]       = -(self.act(self.x_[self.varmap['cAMP']],rpar[:,51])*xmax[self.varmap['PKA']] - self.x_[self.varmap['PKA']])/tau[self.varmap['PKA']]
        self.f_[self.varmap['PKC']]       = -(self.OR(self.act(self.x_[self.varmap['Calcium']],rpar[:,66])*self.act(self.x_[self.varmap['DAG']],rpar[:,66]),self.act(self.x_[self.varmap['TGFR']],rpar[:,189]))*xmax[self.varmap['PKC']] - self.x_[self.varmap['PKC']])/tau[self.varmap['PKC']]
        self.f_[self.varmap['PKD']]       = -(self.act(self.x_[self.varmap['PKC']],rpar[:,164])*xmax[self.varmap['PKD']] - self.x_[self.varmap['PKD']])/tau[self.varmap['PKD']]
        self.f_[self.varmap['PKG1']]      = -(self.act(self.x_[self.varmap['cGMP']],rpar[:,57])*xmax[self.varmap['PKG1']] - self.x_[self.varmap['PKG1']])/tau[self.varmap['PKG1']]
        self.f_[self.varmap['PLCB']]      = -(self.OR(self.act(self.x_[self.varmap['Gaq11']],rpar[:,89]),self.act(self.x_[self.varmap['IGF1R']],rpar[:,103]))*xmax[self.varmap['PLCB']] - self.x_[self.varmap['PLCB']])/tau[self.varmap['PLCB']]
        self.f_[self.varmap['PLCG']]      = -(self.OR(self.act(self.x_[self.varmap['EGFR']],rpar[:,69]),self.act(self.x_[self.varmap['ERBB']],rpar[:,73]))*xmax[self.varmap['PLCG']] - self.x_[self.varmap['PLCG']])/tau[self.varmap['PLCG']]
        self.f_[self.varmap['Rac1']]      = -(self.act(self.x_[self.varmap['Ras']],rpar[:,179])*xmax[self.varmap['Rac1']] - self.x_[self.varmap['Rac1']])/tau[self.varmap['Rac1']]
        self.f_[self.varmap['Raf1']]      = -(self.inhib(self.x_[self.varmap['PKA']],rpar[:,30])*self.act(self.x_[self.varmap['Raf1A']],rpar[:,30])*xmax[self.varmap['Raf1']] - self.x_[self.varmap['Raf1']])/tau[self.varmap['Raf1']]
        self.f_[self.varmap['Raf1A']]     = -(self.OR(self.act(self.x_[self.varmap['GBG']],rpar[:,94]),self.OR(self.act(self.x_[self.varmap['PKC']],rpar[:,165]),self.act(self.x_[self.varmap['Ras']],rpar[:,180])))*xmax[self.varmap['Raf1A']] - self.x_[self.varmap['Raf1A']])/tau[self.varmap['Raf1A']]
        self.f_[self.varmap['Ras']]       = -(self.OR(self.act(self.x_[self.varmap['EGFR']],rpar[:,70]),self.OR(self.act(self.x_[self.varmap['ERBB']],rpar[:,74]),self.OR(self.act(self.x_[self.varmap['FAK']],rpar[:,85]),self.OR(self.act(self.x_[self.varmap['FGFR']],rpar[:,87]),self.OR(self.act(self.x_[self.varmap['GBG']],rpar[:,95]),self.OR(self.act(self.x_[self.varmap['IGF1R']],rpar[:,104]),self.OR(self.act(self.x_[self.varmap['JAK']],rpar[:,111]),self.act(self.x_[self.varmap['PKC']],rpar[:,166]))))))))*xmax[self.varmap['Ras']] - self.x_[self.varmap['Ras']])/tau[self.varmap['Ras']]
        self.f_[self.varmap['RhoA']]      = -(self.act(self.x_[self.varmap['Ras']],rpar[:,175])*self.inhib(self.x_[self.varmap['SHP2']],rpar[:,175])*xmax[self.varmap['RhoA']] - self.x_[self.varmap['RhoA']])/tau[self.varmap['RhoA']]
        self.f_[self.varmap['sACT']]      = -(self.OR(self.act(self.x_[self.varmap['cFos']],rpar[:,53])*self.act(self.x_[self.varmap['cJun']],rpar[:,53])*self.act(self.x_[self.varmap['SRF']],rpar[:,53]),self.OR(self.act(self.x_[self.varmap['cJun']],rpar[:,58])*self.act(self.x_[self.varmap['SRF']],rpar[:,58]),self.OR(self.act(self.x_[self.varmap['GATA4']],rpar[:,90])*self.act(self.x_[self.varmap['SRF']],rpar[:,90]),self.OR(self.act(self.x_[self.varmap['MEF2']],rpar[:,130]),self.act(self.x_[self.varmap['NFAT']],rpar[:,148])))))*xmax[self.varmap['sACT']] - self.x_[self.varmap['sACT']])/tau[self.varmap['sACT']]
        self.f_[self.varmap['SERCA']]     = -(self.inhib(self.x_[self.varmap['cFos']],rpar[:,21])*self.inhib(self.x_[self.varmap['cJun']],rpar[:,21])*self.inhib(self.x_[self.varmap['NFAT']],rpar[:,21])*xmax[self.varmap['SERCA']] - self.x_[self.varmap['SERCA']])/tau[self.varmap['SERCA']]
        self.f_[self.varmap['sGC']]       = -(self.act(self.x_[self.varmap['NOS']],rpar[:,150])*xmax[self.varmap['sGC']] - self.x_[self.varmap['sGC']])/tau[self.varmap['sGC']]
        self.f_[self.varmap['SHP2']]      = -(self.act(self.x_[self.varmap['gp130LIFR']],rpar[:,98])*xmax[self.varmap['SHP2']] - self.x_[self.varmap['SHP2']])/tau[self.varmap['SHP2']]
        self.f_[self.varmap['SRF']]       = -(self.act(self.x_[self.varmap['RhoA']],rpar[:,181])*xmax[self.varmap['SRF']] - self.x_[self.varmap['SRF']])/tau[self.varmap['SRF']]
        self.f_[self.varmap['STAT']]      = -(self.act(self.x_[self.varmap['JAK']],rpar[:,112])*xmax[self.varmap['STAT']] - self.x_[self.varmap['STAT']])/tau[self.varmap['STAT']]
        self.f_[self.varmap['Stretch']]   = -(rpar[0,14]*xmax[self.varmap['Stretch']] - self.x_[self.varmap['Stretch']])/tau[self.varmap['Stretch']]
        self.f_[self.varmap['TAK1']]      = -(self.act(self.x_[self.varmap['PKC']],rpar[:,167])*xmax[self.varmap['TAK1']] - self.x_[self.varmap['TAK1']])/tau[self.varmap['TAK1']]
        self.f_[self.varmap['TGFB']]      = -(rpar[0,15]*xmax[self.varmap['TGFB']] - self.x_[self.varmap['TGFB']])/tau[self.varmap['TGFB']]
        self.f_[self.varmap['TGFR']]      = -(self.act(self.x_[self.varmap['TGFB']],rpar[:,188])*xmax[self.varmap['TGFR']] - self.x_[self.varmap['TGFR']])/tau[self.varmap['TGFR']]
        self.f_[self.varmap['TNFa']]      = -(rpar[0,16]*xmax[self.varmap['TNFa']] - self.x_[self.varmap['TNFa']])/tau[self.varmap['TNFa']]
        self.f_[self.varmap['TNFR']]      = -(self.act(self.x_[self.varmap['TNFa']],rpar[:,190])*xmax[self.varmap['TNFR']] - self.x_[self.varmap['TNFR']])/tau[self.varmap['TNFR']]


    def act(self, x, rp):

        # Hill activation function with parameters w (weight), n (Hill coefficient), EC50
        w = rp[0]
        n = rp[1]
        EC50 = rp[2]

        beta = (EC50**n - 1.)/(2.*EC50**n - 1.)
        K = (beta - 1.)**(1./n)

        fact = sp.Min(w*(beta*x**n)/(K**n + x**n), w)

        return fact


    def inhib(self, x, rp):

        return rp[0] - self.act(x,rp)


    def OR(self, a, b):

        return a + b - a*b


    def initialize(self, var, iniparam):

        for k in self.varmap.keys():
            try: var[self.varmap[k]] = iniparam[k+'_0']
            except: var[self.varmap[k]] = 0.0


    # print during simulation
    def print_to_screen(self, var, aux):

        if isinstance(var, np.ndarray): var_sq = var
        else: var_sq = allgather_vec(var, self.comm)

        pass

        if not isinstance(var, np.ndarray): del var_sq
