#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import sys, math
import numpy as np
import sympy as sp

from .cardiovascular0D_syspulcap import cardiovascular0Dsyspulcap
from ..mpiroutines import allgather_vec

# respiratory and gas transport part of syspulcap model (Diss Hirschvogel, p. 58ff.)
# builds upon syspulcap model

class cardiovascular0Dsyspulcaprespir(cardiovascular0Dsyspulcap):

    def __init__(self, params, chmodels, cq, vq, valvelaws={'av' : ['pwlin_pres',0], 'mv' : ['pwlin_pres',0], 'pv' : ['pwlin_pres',0], 'tv' : ['pwlin_pres',0]}, cormodel=None, vadmodel=None, init=True, comm=None):

        self.R_airw = params['R_airw']
        self.L_alv = params['L_alv']
        self.R_alv = params['R_alv']
        self.E_alv = params['E_alv']

        self.U_m = params['U_m']

        self.V_lung_dead = params['V_lung_dead'] # dead space volume
        self.V_lung_u = params['V_lung_u'] # unstressed lung volume (volume of the lung when it is fully collapsed outside the body)
        self.V_lung_total = params['V_lung_total']

        self.V_lung_tidal = params['V_lung_tidal'] # tidal volume (the total volume of inspired air, in a single breath)

        self.omega_breath = params['omega_breath'] # unstressed lung volume (volume of the lung when it is fully collapsed outside the body)

        self.fCO2_ext = params['fCO2_ext']
        self.fO2_ext = params['fO2_ext']

        self.V_m_gas = params['V_m_gas'] # molar volume of an ideal gas

        self.p_vap_water_37 = params['p_vap_water_37'] # vapor pressure of water at 37 °C

        self.kappa_CO2 = params['kappa_CO2'] # diffusion coefficient for CO2 across the hemato-alveolar membrane, in molar value / (time * pressure)
        self.kappa_O2 = params['kappa_O2'] # diffusion coefficient for CO2 across the hemato-alveolar membrane, in molar value / (time * pressure)

        self.alpha_CO2 = params['alpha_CO2'] # CO2 solubility constant, in molar value / (volume * pressure)
        self.alpha_O2 = params['alpha_O2'] # O2 solubility constant, in molar value / (volume * pressure)

        self.c_Hb = params['c_Hb'] # hemoglobin concentration of the blood, in molar value / volume (default: Christiansen (1996), p. 92, unit: mmol/mm^3)

        self.M_CO2_arspl = params['M_CO2_arspl'] # splanchnic metabolic rate of CO2 production
        self.M_O2_arspl = params['M_O2_arspl'] # splanchnic metabolic rate of O2 consumption
        self.M_CO2_arespl = params['M_CO2_arespl'] # extra-splanchnic metabolic rate of CO2 production
        self.M_O2_arespl = params['M_O2_arespl'] # extra-splanchnic metabolic rate of O2 consumption
        self.M_CO2_armsc = params['M_CO2_armsc'] # muscular metabolic rate of CO2 production
        self.M_O2_armsc = params['M_O2_armsc'] # muscular metabolic rate of O2 consumption
        self.M_CO2_arcer = params['M_CO2_arcer'] # cerebral metabolic rate of CO2 production
        self.M_O2_arcer = params['M_O2_arcer'] # cerebral metabolic rate of O2 consumption
        self.M_CO2_arcor = params['M_CO2_arcor'] # coronary metabolic rate of CO2 production
        self.M_O2_arcor = params['M_O2_arcor'] # coronary metabolic rate of O2 consumption

        self.beta_O2 = params['beta_O2'] # oxygen concentration when the metabolic rate is half of the maximum value (Christiansen (1996), p. 52)

        self.V_tissspl = params['V_tissspl']
        self.V_tissespl = params['V_tissespl']
        self.V_tissmsc = params['V_tissmsc']
        self.V_tisscer = params['V_tisscer']
        self.V_tisscor = params['V_tisscor']

        # initialize base class
        super().__init__(params, chmodels, cq, vq, valvelaws, cormodel=cormodel, vadmodel=vadmodel, init=init, comm=comm)


    def setup_arrays(self):

        # number of degrees of freedom
        self.numdof = 84

        self.elastarrays = [[]]*4

        self.si, self.switch_V = [0]*5, [1]*5 # default values

        self.varindex_ch = [3,29,1,27] # coupling variable indices (decreased by 1 for pressure coupling!)
        self.vname, self.cname = ['p_v_l','p_v_r','p_at_l','p_at_r'], []

        self.set_solve_arrays()


    def equation_map(self):

        cardiovascular0Dsyspulcap.equation_map(self)

        # add to varmap
        self.varmap['V_alv']             = 36
        self.varmap['q_alv']             = 37
        self.varmap['p_alv']             = 38
        self.varmap['fCO2_alv']          = 39
        self.varmap['fO2_alv']           = 40
        self.varmap['q_arspl_sys_in']    = 41
        self.varmap['q_arespl_sys_in']   = 42
        self.varmap['q_armsc_sys_in']    = 43
        self.varmap['q_arcer_sys_in']    = 44
        self.varmap['q_arcor_sys_in']    = 45
        self.varmap['ppCO2_at_r']        = 46
        self.varmap['ppO2_at_r']         = 47
        self.varmap['ppCO2_v_r']         = 48
        self.varmap['ppO2_v_r']          = 49
        self.varmap['ppCO2_ar_pul']      = 50
        self.varmap['ppO2_ar_pul']       = 51
        self.varmap['ppCO2_cap_pul']     = 52
        self.varmap['ppO2_cap_pul']      = 53
        self.varmap['ppCO2_ven_pul']     = 54
        self.varmap['ppO2_ven_pul']      = 55
        self.varmap['ppCO2_at_l']        = 56
        self.varmap['ppO2_at_l']         = 57
        self.varmap['ppCO2_v_l']         = 58
        self.varmap['ppO2_v_l']          = 59
        self.varmap['ppCO2_ar_sys']      = 60
        self.varmap['ppO2_ar_sys']       = 61
        self.varmap['ppCO2_arspl_sys']   = 62
        self.varmap['ppO2_arspl_sys']    = 63
        self.varmap['ppCO2_arespl_sys']  = 64
        self.varmap['ppO2_arespl_sys']   = 65
        self.varmap['ppCO2_armsc_sys']   = 66
        self.varmap['ppO2_armsc_sys']    = 67
        self.varmap['ppCO2_arcer_sys']   = 68
        self.varmap['ppO2_arcer_sys']    = 69
        self.varmap['ppCO2_arcor_sys']   = 70
        self.varmap['ppO2_arcor_sys']    = 71
        self.varmap['ppCO2_venspl_sys']  = 72
        self.varmap['ppO2_venspl_sys']   = 73
        self.varmap['ppCO2_venespl_sys'] = 74
        self.varmap['ppO2_venespl_sys']  = 75
        self.varmap['ppCO2_venmsc_sys']  = 76
        self.varmap['ppO2_venmsc_sys']   = 77
        self.varmap['ppCO2_vencer_sys']  = 78
        self.varmap['ppO2_vencer_sys']   = 79
        self.varmap['ppCO2_vencor_sys']  = 80
        self.varmap['ppO2_vencor_sys']   = 81
        self.varmap['ppCO2_ven_sys']     = 82
        self.varmap['ppO2_ven_sys']      = 83

        # add to auxmap
        self.auxmap['SO2_ar_pul'] = 51
        self.auxmap['SO2_ar_sys'] = 61

        # variables from the mechanics model
        q_vin_l_       = self.x_[0]
        p_at_l_        = self.x_[1]
        q_vout_l_      = self.x_[2]
        p_v_l_         = self.x_[3]
        q_ar_sys_      = self.x_[7]
        p_arperi_sys_  = self.x_[8]
        q_arspl_sys_   = self.x_[9]
        q_arespl_sys_  = self.x_[10]
        q_armsc_sys_   = self.x_[11]
        q_arcer_sys_   = self.x_[12]
        q_arcor_sys_   = self.x_[13]
        q_venspl_sys_  = self.x_[15]
        q_venespl_sys_ = self.x_[17]
        q_venmsc_sys_  = self.x_[19]
        q_vencer_sys_  = self.x_[21]
        q_vencor_sys_  = self.x_[23]
        q_ven_sys_     = self.x_[25]
        q_vin_r_       = self.x_[26]
        p_at_r_        = self.x_[27]
        q_vout_r_      = self.x_[28]
        p_v_r_         = self.x_[29]
        q_ar_pul_      = self.x_[31]
        q_cap_pul_     = self.x_[33]
        q_ven_pul_     = self.x_[35]
        # volumes from the mechanics model
        V_arspl_sys_   = self.C_arspl_sys * p_arperi_sys_ + self.V_arspl_sys_u
        V_arespl_sys_  = self.C_arespl_sys * p_arperi_sys_ + self.V_arespl_sys_u
        V_armsc_sys_   = self.C_armsc_sys * p_arperi_sys_ + self.V_armsc_sys_u
        V_arcer_sys_   = self.C_arcer_sys * p_arperi_sys_ + self.V_arcer_sys_u
        V_arcor_sys_   = self.C_arcor_sys * p_arperi_sys_ + self.V_arcor_sys_u

        V_at_l_        = self.a_[self.auxmap['V_at_l']]
        V_v_l_         = self.a_[self.auxmap['V_v_l']]
        V_ar_sys_      = self.a_[self.auxmap['V_ar_sys']]
        V_arperi_sys_  = self.a_[self.auxmap['V_arperi_sys']]
        V_venspl_sys_  = self.a_[self.auxmap['V_venspl_sys']]
        V_venespl_sys_ = self.a_[self.auxmap['V_venespl_sys']]
        V_venmsc_sys_  = self.a_[self.auxmap['V_venmsc_sys']]
        V_vencer_sys_  = self.a_[self.auxmap['V_vencer_sys']]
        V_vencor_sys_  = self.a_[self.auxmap['V_vencor_sys']]
        V_ven_sys_     = self.a_[self.auxmap['V_ven_sys']]
        V_at_r_        = self.a_[self.auxmap['V_at_r']]
        V_v_r_         = self.a_[self.auxmap['V_v_r']]
        V_ar_pul_      = self.a_[self.auxmap['V_ar_pul']]
        V_cap_pul_     = self.a_[self.auxmap['V_cap_pul']]
        V_ven_pul_     = self.a_[self.auxmap['V_ven_pul']]

        # respiratory model variables
        V_alv_ = sp.Symbol('V_alv_')
        q_alv_ = sp.Symbol('q_alv_')
        p_alv_ = sp.Symbol('p_alv_')
        fCO2_alv_ = sp.Symbol('fCO2_alv_')
        fO2_alv_ = sp.Symbol('fO2_alv_')

        q_arspl_sys_in_ = sp.Symbol('q_arspl_sys_in_')
        q_arespl_sys_in_ = sp.Symbol('q_arespl_sys_in_')
        q_armsc_sys_in_ = sp.Symbol('q_armsc_sys_in_')
        q_arcer_sys_in_ = sp.Symbol('q_arcer_sys_in_')
        q_arcor_sys_in_ = sp.Symbol('q_arcor_sys_in_')

        ppCO2_at_r_ = sp.Symbol('ppCO2_at_r_')
        ppO2_at_r_ = sp.Symbol('ppO2_at_r_')
        ppCO2_v_r_ = sp.Symbol('ppCO2_v_r_')
        ppO2_v_r_ = sp.Symbol('ppO2_v_r_')
        ppCO2_ar_pul_ = sp.Symbol('ppCO2_ar_pul_')
        ppO2_ar_pul_ = sp.Symbol('ppO2_ar_pul_')
        # gas partial pressures at pulmonary capillaries
        ppCO2_cap_pul_ = sp.Symbol('ppCO2_cap_pul_')
        ppO2_cap_pul_ = sp.Symbol('ppO2_cap_pul_')

        ppCO2_ven_pul_ = sp.Symbol('ppCO2_ven_pul_')
        ppO2_ven_pul_ = sp.Symbol('ppO2_ven_pul_')
        ppCO2_at_l_ = sp.Symbol('ppCO2_at_l_')
        ppO2_at_l_ = sp.Symbol('ppO2_at_l_')
        ppCO2_v_l_ = sp.Symbol('ppCO2_v_l_')
        ppO2_v_l_ = sp.Symbol('ppO2_v_l_')
        ppCO2_ar_sys_ = sp.Symbol('ppCO2_ar_sys_')
        ppO2_ar_sys_ = sp.Symbol('ppO2_ar_sys_')

        # gas partial pressures at systemic capillaries
        ppCO2_arspl_sys_ = sp.Symbol('ppCO2_arspl_sys_')
        ppO2_arspl_sys_ = sp.Symbol('ppO2_arspl_sys_')
        ppCO2_arespl_sys_ = sp.Symbol('ppCO2_arespl_sys_')
        ppO2_arespl_sys_ = sp.Symbol('ppO2_arespl_sys_')
        ppCO2_armsc_sys_ = sp.Symbol('ppCO2_armsc_sys_')
        ppO2_armsc_sys_ = sp.Symbol('ppO2_armsc_sys_')
        ppCO2_arcer_sys_ = sp.Symbol('ppCO2_arcer_sys_')
        ppO2_arcer_sys_ = sp.Symbol('ppO2_arcer_sys_')
        ppCO2_arcor_sys_ = sp.Symbol('ppCO2_arcor_sys_')
        ppO2_arcor_sys_ = sp.Symbol('ppO2_arcor_sys_')

        ppCO2_venspl_sys_ = sp.Symbol('ppCO2_venspl_sys_')
        ppO2_venspl_sys_ = sp.Symbol('ppO2_venspl_sys_')
        ppCO2_venespl_sys_ = sp.Symbol('ppCO2_venespl_sys_')
        ppO2_venespl_sys_ = sp.Symbol('ppO2_venespl_sys_')
        ppCO2_venmsc_sys_ = sp.Symbol('ppCO2_venmsc_sys_')
        ppO2_venmsc_sys_ = sp.Symbol('ppO2_venmsc_sys_')
        ppCO2_vencer_sys_ = sp.Symbol('ppCO2_vencer_sys_')
        ppO2_vencer_sys_ = sp.Symbol('ppO2_vencer_sys_')
        ppCO2_vencor_sys_ = sp.Symbol('ppCO2_vencor_sys_')
        ppO2_vencor_sys_ = sp.Symbol('ppO2_vencor_sys_')
        ppCO2_ven_sys_ = sp.Symbol('ppCO2_ven_sys_')
        ppO2_ven_sys_ = sp.Symbol('ppO2_ven_sys_')

        self.x_[36] = V_alv_
        self.x_[37] = q_alv_
        self.x_[38] = p_alv_
        self.x_[39] = fCO2_alv_
        self.x_[40] = fO2_alv_
        self.x_[41] = q_arspl_sys_in_
        self.x_[42] = q_arespl_sys_in_
        self.x_[43] = q_armsc_sys_in_
        self.x_[44] = q_arcer_sys_in_
        self.x_[45] = q_arcor_sys_in_
        self.x_[46] = ppCO2_at_r_
        self.x_[47] = ppO2_at_r_
        self.x_[48] = ppCO2_v_r_
        self.x_[49] = ppO2_v_r_
        self.x_[50] = ppCO2_ar_pul_
        self.x_[51] = ppO2_ar_pul_
        self.x_[52] = ppCO2_cap_pul_
        self.x_[53] = ppO2_cap_pul_
        self.x_[54] = ppCO2_ven_pul_
        self.x_[55] = ppO2_ven_pul_
        self.x_[56] = ppCO2_at_l_
        self.x_[57] = ppO2_at_l_
        self.x_[58] = ppCO2_v_l_
        self.x_[59] = ppO2_v_l_
        self.x_[60] = ppCO2_ar_sys_
        self.x_[61] = ppO2_ar_sys_
        self.x_[62] = ppCO2_arspl_sys_
        self.x_[63] = ppO2_arspl_sys_
        self.x_[64] = ppCO2_arespl_sys_
        self.x_[65] = ppO2_arespl_sys_
        self.x_[66] = ppCO2_armsc_sys_
        self.x_[67] = ppO2_armsc_sys_
        self.x_[68] = ppCO2_arcer_sys_
        self.x_[69] = ppO2_arcer_sys_
        self.x_[70] = ppCO2_arcor_sys_
        self.x_[71] = ppO2_arcor_sys_
        self.x_[72] = ppCO2_venspl_sys_
        self.x_[73] = ppO2_venspl_sys_
        self.x_[74] = ppCO2_venespl_sys_
        self.x_[75] = ppO2_venespl_sys_
        self.x_[76] = ppCO2_venmsc_sys_
        self.x_[77] = ppO2_venmsc_sys_
        self.x_[78] = ppCO2_vencer_sys_
        self.x_[79] = ppO2_vencer_sys_
        self.x_[80] = ppCO2_vencor_sys_
        self.x_[81] = ppO2_vencor_sys_
        self.x_[82] = ppCO2_ven_sys_
        self.x_[83] = ppO2_ven_sys_

        # 0D lung
        self.df_[36] = V_alv_
        self.df_[37] = self.L_alv * q_alv_
        self.df_[38] = p_alv_

        fCO2_insp_ = sp.Piecewise( ((fCO2_alv_ * self.V_lung_dead + self.fCO2_ext * (self.V_lung_tidal-self.V_lung_dead)) / self.V_lung_tidal, self.V_lung_tidal >= self.V_lung_dead), (fCO2_alv_, self.V_lung_tidal < self.V_lung_dead) )
        fO2_insp_  = sp.Piecewise( ((fO2_alv_ * self.V_lung_dead + self.fO2_ext * (self.V_lung_tidal-self.V_lung_dead)) / self.V_lung_tidal, self.V_lung_tidal >= self.V_lung_dead), (fO2_alv_, self.V_lung_tidal < self.V_lung_dead) )

        q_insp_ = sp.Piecewise( ((self.U_m-p_alv_)/self.R_airw, self.U_m > p_alv_), (0, self.U_m <= p_alv_) )

        self.df_[39] = fCO2_alv_
        self.df_[40] = fO2_alv_

        self.df_[41] = self.C_arspl_sys * p_arperi_sys_
        self.df_[42] = self.C_arespl_sys * p_arperi_sys_
        self.df_[43] = self.C_armsc_sys * p_arperi_sys_
        self.df_[44] = self.C_arcer_sys * p_arperi_sys_
        self.df_[45] = self.C_arcor_sys * p_arperi_sys_

        # gas transport in cardiovascular system
        self.df_[46] = ppCO2_at_r_
        self.df_[47] = ppO2_at_r_
        self.df_[48] = ppCO2_v_r_
        self.df_[49] = ppO2_v_r_
        self.df_[50] = ppCO2_ar_pul_
        self.df_[51] = ppO2_ar_pul_

        # gas partial pressures at pulmonary capillaries
        self.df_[52] = ppCO2_cap_pul_
        self.df_[53] = ppO2_cap_pul_

        self.df_[54] = ppCO2_ven_pul_
        self.df_[55] = ppO2_ven_pul_
        self.df_[56] = ppCO2_at_l_
        self.df_[57] = ppO2_at_l_
        self.df_[58] = ppCO2_v_l_
        self.df_[59] = ppO2_v_l_
        self.df_[60] = ppCO2_ar_sys_
        self.df_[61] = ppO2_ar_sys_

        # gas partial pressures at systemic capillaries
        # arterioles
        self.df_[62] = ppCO2_arspl_sys_
        self.df_[63] = ppO2_arspl_sys_
        self.df_[64] = ppCO2_arespl_sys_
        self.df_[65] = ppO2_arespl_sys_
        self.df_[66] = ppCO2_armsc_sys_
        self.df_[67] = ppO2_armsc_sys_
        self.df_[68] = ppCO2_arcer_sys_
        self.df_[69] = ppO2_arcer_sys_
        self.df_[70] = ppCO2_arcor_sys_
        self.df_[71] = ppO2_arcor_sys_
        # venules
        self.df_[72] = ppCO2_venspl_sys_
        self.df_[73] = ppO2_venspl_sys_
        self.df_[74] = ppCO2_venespl_sys_
        self.df_[75] = ppO2_venespl_sys_
        self.df_[76] = ppCO2_venmsc_sys_
        self.df_[77] = ppO2_venmsc_sys_
        self.df_[78] = ppCO2_vencer_sys_
        self.df_[79] = ppO2_vencer_sys_
        self.df_[80] = ppCO2_vencor_sys_
        self.df_[81] = ppO2_vencor_sys_
        self.df_[82] = ppCO2_ven_sys_
        self.df_[83] = ppO2_ven_sys_

        self.f_[36] = -q_alv_
        self.f_[37] = self.R_alv * q_alv_ + self.E_alv*(V_alv_-self.V_lung_u) - p_alv_ + self.U_t()
        self.f_[38] = -(1./V_alv_) * (self.U_m * ((self.U_m-p_alv_)/self.R_airw + self.V_m_gas*self.kappa_CO2*(ppCO2_cap_pul_ - fCO2_alv_*(p_alv_-self.p_vap_water_37)) + self.V_m_gas*self.kappa_O2*(ppO2_cap_pul_ - fO2_alv_*(p_alv_-self.p_vap_water_37))) - p_alv_ * q_alv_)

        self.f_[39] = -(1./V_alv_) * ( self.V_m_gas*self.kappa_CO2*(ppCO2_cap_pul_ - fCO2_alv_*p_alv_) + (fCO2_insp_ - fCO2_alv_)*q_insp_ - fCO2_alv_*(self.V_m_gas*self.kappa_O2*(ppO2_cap_pul_ - fO2_alv_*(p_alv_-self.p_vap_water_37)) + self.V_m_gas*self.kappa_CO2*(ppCO2_cap_pul_ - fCO2_alv_*(p_alv_-self.p_vap_water_37))))
        self.f_[40] = -(1./V_alv_) * ( self.V_m_gas*self.kappa_O2*(ppO2_cap_pul_ - fO2_alv_*p_alv_) + (fO2_insp_ - fO2_alv_)*q_insp_ - fO2_alv_*(self.V_m_gas*self.kappa_CO2*(ppCO2_cap_pul_ - fCO2_alv_*(p_alv_-self.p_vap_water_37)) + self.V_m_gas*self.kappa_O2*(ppO2_cap_pul_ - fO2_alv_*(p_alv_-self.p_vap_water_37))))

        self.f_[41] = q_arspl_sys_ - q_arspl_sys_in_
        self.f_[42] = q_arespl_sys_ - q_arespl_sys_in_
        self.f_[43] = q_armsc_sys_ - q_armsc_sys_in_
        self.f_[44] = q_arcer_sys_ - q_arcer_sys_in_
        self.f_[45] = q_arcor_sys_ - q_arcor_sys_in_

        # right atrium CO2
        self.f_[46] = (1./V_at_r_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_at_r_,ppO2_at_r_)*self.dcbO2_dppO2(ppCO2_at_r_,ppO2_at_r_) - self.dcbO2_dppCO2(ppCO2_at_r_,ppO2_at_r_)*self.dcbCO2_dppO2(ppCO2_at_r_,ppO2_at_r_) ),-1.) * \
            ( self.dcbO2_dppO2(ppCO2_at_r_,ppO2_at_r_) * (q_ven_sys_ * (self.cbCO2(ppCO2_at_r_,ppO2_at_r_) - self.cbCO2(ppCO2_ven_sys_,ppO2_ven_sys_))) - \
                self.dcbCO2_dppO2(ppCO2_at_r_,ppO2_at_r_) * (q_ven_sys_ * (self.cbO2(ppCO2_at_r_,ppO2_at_r_) - self.cbO2(ppCO2_ven_sys_,ppO2_ven_sys_))) )
        # right atrium O2
        self.f_[47] = (1./V_at_r_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_at_r_,ppO2_at_r_)*self.dcbO2_dppO2(ppCO2_at_r_,ppO2_at_r_) - self.dcbO2_dppCO2(ppCO2_at_r_,ppO2_at_r_)*self.dcbCO2_dppO2(ppCO2_at_r_,ppO2_at_r_) ),-1.) * \
            ( self.dcbCO2_dppCO2(ppCO2_at_r_,ppO2_at_r_) * (q_ven_sys_ * (self.cbO2(ppCO2_at_r_,ppO2_at_r_) - self.cbO2(ppCO2_ven_sys_,ppO2_ven_sys_))) - \
                self.dcbO2_dppCO2(ppCO2_at_r_,ppO2_at_r_) * (q_ven_sys_ * (self.cbCO2(ppCO2_at_r_,ppO2_at_r_) - self.cbCO2(ppCO2_ven_sys_,ppO2_ven_sys_))) )

        # right ventricle CO2
        self.f_[48] = (1./V_v_r_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_v_r_,ppO2_v_r_)*self.dcbO2_dppO2(ppCO2_v_r_,ppO2_v_r_) - self.dcbO2_dppCO2(ppCO2_v_r_,ppO2_v_r_)*self.dcbCO2_dppO2(ppCO2_v_r_,ppO2_v_r_) ),-1.) * \
            ( self.dcbO2_dppO2(ppCO2_v_r_,ppO2_v_r_) * (q_vin_r_ * (self.cbCO2(ppCO2_v_r_,ppO2_v_r_) - self.cbCO2(ppCO2_at_r_,ppO2_at_r_))) - \
                self.dcbCO2_dppO2(ppCO2_v_r_,ppO2_v_r_) * (q_vin_r_ * (self.cbO2(ppCO2_v_r_,ppO2_v_r_) - self.cbO2(ppCO2_at_r_,ppO2_at_r_))) )
        # right ventricle O2
        self.f_[49] = (1./V_v_r_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_v_r_,ppO2_v_r_)*self.dcbO2_dppO2(ppCO2_v_r_,ppO2_v_r_) - self.dcbO2_dppCO2(ppCO2_v_r_,ppO2_v_r_)*self.dcbCO2_dppO2(ppCO2_v_r_,ppO2_v_r_) ),-1.) * \
            ( self.dcbCO2_dppCO2(ppCO2_v_r_,ppO2_v_r_) * (q_vin_r_ * (self.cbO2(ppCO2_v_r_,ppO2_v_r_) - self.cbO2(ppCO2_at_r_,ppO2_at_r_))) - \
                self.dcbO2_dppCO2(ppCO2_v_r_,ppO2_v_r_) * (q_vin_r_ * (self.cbCO2(ppCO2_v_r_,ppO2_v_r_) - self.cbCO2(ppCO2_at_r_,ppO2_at_r_))) )

        # pulmonary arteries CO2
        self.f_[50] = (1./V_ar_pul_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_ar_pul_,ppO2_ar_pul_)*self.dcbO2_dppO2(ppCO2_ar_pul_,ppO2_ar_pul_) - self.dcbO2_dppCO2(ppCO2_ar_pul_,ppO2_ar_pul_)*self.dcbCO2_dppO2(ppCO2_ar_pul_,ppO2_ar_pul_) ),-1.) * \
            ( self.dcbO2_dppO2(ppCO2_ar_pul_,ppO2_ar_pul_) * (q_vout_r_ * (self.cbCO2(ppCO2_ar_pul_,ppO2_ar_pul_) - self.cbCO2(ppCO2_v_r_,ppO2_v_r_))) - \
                self.dcbCO2_dppO2(ppCO2_ar_pul_,ppO2_ar_pul_) * (q_vout_r_ * (self.cbO2(ppCO2_ar_pul_,ppO2_ar_pul_) - self.cbO2(ppCO2_v_r_,ppO2_v_r_))) )
        # pulmonary arteries O2
        self.f_[51] = (1./V_ar_pul_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_ar_pul_,ppO2_ar_pul_)*self.dcbO2_dppO2(ppCO2_ar_pul_,ppO2_ar_pul_) - self.dcbO2_dppCO2(ppCO2_ar_pul_,ppO2_ar_pul_)*self.dcbCO2_dppO2(ppCO2_ar_pul_,ppO2_ar_pul_) ),-1.) * \
            ( self.dcbCO2_dppCO2(ppCO2_ar_pul_,ppO2_ar_pul_) * (q_vout_r_ * (self.cbO2(ppCO2_ar_pul_,ppO2_ar_pul_) - self.cbO2(ppCO2_v_r_,ppO2_v_r_))) - \
                self.dcbO2_dppCO2(ppCO2_ar_pul_,ppO2_ar_pul_) * (q_vout_r_ * (self.cbCO2(ppCO2_ar_pul_,ppO2_ar_pul_) - self.cbCO2(ppCO2_v_r_,ppO2_v_r_))) )

        # pulmonary capillaries CO2
        self.f_[52] = (1./V_cap_pul_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_cap_pul_,ppO2_cap_pul_)*self.dcbO2_dppO2(ppCO2_cap_pul_,ppO2_cap_pul_) - self.dcbO2_dppCO2(ppCO2_cap_pul_,ppO2_cap_pul_)*self.dcbCO2_dppO2(ppCO2_cap_pul_,ppO2_cap_pul_) ),-1.) * \
            ( self.dcbO2_dppO2(ppCO2_cap_pul_,ppO2_cap_pul_) * (q_ar_pul_ * (self.cbCO2(ppCO2_cap_pul_,ppO2_cap_pul_) - self.cbCO2(ppCO2_ar_pul_,ppO2_ar_pul_)) + self.kappa_CO2*(ppCO2_cap_pul_ - fCO2_alv_*(p_alv_-self.p_vap_water_37))) - \
                self.dcbCO2_dppO2(ppCO2_cap_pul_,ppO2_cap_pul_) * (q_ar_pul_ * (self.cbO2(ppCO2_cap_pul_,ppO2_cap_pul_) - self.cbO2(ppCO2_ar_pul_,ppO2_ar_pul_)) + self.kappa_O2*(ppO2_cap_pul_ - fO2_alv_*(p_alv_-self.p_vap_water_37))) )
        # pulmonary capillaries O2
        self.f_[53] = (1./V_cap_pul_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_cap_pul_,ppO2_cap_pul_)*self.dcbO2_dppO2(ppCO2_cap_pul_,ppO2_cap_pul_) - self.dcbO2_dppCO2(ppCO2_cap_pul_,ppO2_cap_pul_)*self.dcbCO2_dppO2(ppCO2_cap_pul_,ppO2_cap_pul_) ),-1.) * \
            ( self.dcbCO2_dppCO2(ppCO2_cap_pul_,ppO2_cap_pul_) * (q_ar_pul_ * (self.cbO2(ppCO2_cap_pul_,ppO2_cap_pul_) - self.cbO2(ppCO2_ar_pul_,ppO2_ar_pul_)) + self.kappa_O2*(ppO2_cap_pul_ - fO2_alv_*(p_alv_-self.p_vap_water_37))) - \
                self.dcbO2_dppCO2(ppCO2_cap_pul_,ppO2_cap_pul_) * (q_ar_pul_ * (self.cbCO2(ppCO2_cap_pul_,ppO2_cap_pul_) - self.cbCO2(ppCO2_ar_pul_,ppO2_ar_pul_)) + self.kappa_CO2*(ppCO2_cap_pul_ - fCO2_alv_*(p_alv_-self.p_vap_water_37))) )

        # pulmonary veins CO2
        self.f_[54] = (1./V_ven_pul_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_ven_pul_,ppO2_ven_pul_)*self.dcbO2_dppO2(ppCO2_ven_pul_,ppO2_ven_pul_) - self.dcbO2_dppCO2(ppCO2_ven_pul_,ppO2_ven_pul_)*self.dcbCO2_dppO2(ppCO2_ven_pul_,ppO2_ven_pul_) ),-1.) * \
            ( self.dcbO2_dppO2(ppCO2_ven_pul_,ppO2_ven_pul_) * (q_cap_pul_ * (self.cbCO2(ppCO2_ven_pul_,ppO2_ven_pul_) - self.cbCO2(ppCO2_cap_pul_,ppO2_cap_pul_))) - \
                self.dcbCO2_dppO2(ppCO2_ven_pul_,ppO2_ven_pul_) * (q_cap_pul_ * (self.cbO2(ppCO2_ven_pul_,ppO2_ven_pul_) - self.cbO2(ppCO2_cap_pul_,ppO2_cap_pul_))) )
        # pulmonary veins O2
        self.f_[55] = (1./V_ven_pul_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_ven_pul_,ppO2_ven_pul_)*self.dcbO2_dppO2(ppCO2_ven_pul_,ppO2_ven_pul_) - self.dcbO2_dppCO2(ppCO2_ven_pul_,ppO2_ven_pul_)*self.dcbCO2_dppO2(ppCO2_ven_pul_,ppO2_ven_pul_) ),-1.) * \
            ( self.dcbCO2_dppCO2(ppCO2_ven_pul_,ppO2_ven_pul_) * (q_cap_pul_ * (self.cbO2(ppCO2_ven_pul_,ppO2_ven_pul_) - self.cbO2(ppCO2_cap_pul_,ppO2_cap_pul_))) - \
                self.dcbO2_dppCO2(ppCO2_ven_pul_,ppO2_ven_pul_) * (q_cap_pul_ * (self.cbCO2(ppCO2_ven_pul_,ppO2_ven_pul_) - self.cbCO2(ppCO2_cap_pul_,ppO2_cap_pul_))) )

        # left atrium CO2
        self.f_[56] = (1./V_at_l_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_at_l_,ppO2_at_l_)*self.dcbO2_dppO2(ppCO2_at_l_,ppO2_at_l_) - self.dcbO2_dppCO2(ppCO2_at_l_,ppO2_at_l_)*self.dcbCO2_dppO2(ppCO2_at_l_,ppO2_at_l_) ),-1.) * \
            ( self.dcbO2_dppO2(ppCO2_at_l_,ppO2_at_l_) * (q_ven_pul_ * (self.cbCO2(ppCO2_at_l_,ppO2_at_l_) - self.cbCO2(ppCO2_ven_pul_,ppO2_ven_pul_))) - \
                self.dcbCO2_dppO2(ppCO2_at_l_,ppO2_at_l_) * (q_ven_pul_ * (self.cbO2(ppCO2_at_l_,ppO2_at_l_) - self.cbO2(ppCO2_ven_pul_,ppO2_ven_pul_))) )
        # left atrium O2
        self.f_[57] = (1./V_at_l_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_at_l_,ppO2_at_l_)*self.dcbO2_dppO2(ppCO2_at_l_,ppO2_at_l_) - self.dcbO2_dppCO2(ppCO2_at_l_,ppO2_at_l_)*self.dcbCO2_dppO2(ppCO2_at_l_,ppO2_at_l_) ),-1.) * \
            ( self.dcbCO2_dppCO2(ppCO2_at_l_,ppO2_at_l_) * (q_ven_pul_ * (self.cbO2(ppCO2_at_l_,ppO2_at_l_) - self.cbO2(ppCO2_ven_pul_,ppO2_ven_pul_))) - \
                self.dcbO2_dppCO2(ppCO2_at_l_,ppO2_at_l_) * (q_ven_pul_ * (self.cbCO2(ppCO2_at_l_,ppO2_at_l_) - self.cbCO2(ppCO2_ven_pul_,ppO2_ven_pul_))) )

        # left ventricle CO2
        self.f_[58] = (1./V_v_l_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_v_l_,ppO2_v_l_)*self.dcbO2_dppO2(ppCO2_v_l_,ppO2_v_l_) - self.dcbO2_dppCO2(ppCO2_v_l_,ppO2_v_l_)*self.dcbCO2_dppO2(ppCO2_v_l_,ppO2_v_l_) ),-1.) * \
            ( self.dcbO2_dppO2(ppCO2_v_l_,ppO2_v_l_) * (q_vin_l_ * (self.cbCO2(ppCO2_v_l_,ppO2_v_l_) - self.cbCO2(ppCO2_at_l_,ppO2_at_l_))) - \
                self.dcbCO2_dppO2(ppCO2_v_l_,ppO2_v_l_) * (q_vin_l_ * (self.cbO2(ppCO2_v_l_,ppO2_v_l_) - self.cbO2(ppCO2_at_l_,ppO2_at_l_))) )
        # left ventricle O2
        self.f_[59] = (1./V_v_l_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_v_l_,ppO2_v_l_)*self.dcbO2_dppO2(ppCO2_v_l_,ppO2_v_l_) - self.dcbO2_dppCO2(ppCO2_v_l_,ppO2_v_l_)*self.dcbCO2_dppO2(ppCO2_v_l_,ppO2_v_l_) ),-1.) * \
            ( self.dcbCO2_dppCO2(ppCO2_v_l_,ppO2_v_l_) * (q_vin_l_ * (self.cbO2(ppCO2_v_l_,ppO2_v_l_) - self.cbO2(ppCO2_at_l_,ppO2_at_l_))) - \
                self.dcbO2_dppCO2(ppCO2_v_l_,ppO2_v_l_) * (q_vin_l_ * (self.cbCO2(ppCO2_v_l_,ppO2_v_l_) - self.cbCO2(ppCO2_at_l_,ppO2_at_l_))) )

        # systemic arteries CO2
        self.f_[60] = (1./V_ar_sys_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_ar_sys_,ppO2_ar_sys_)*self.dcbO2_dppO2(ppCO2_ar_sys_,ppO2_ar_sys_) - self.dcbO2_dppCO2(ppCO2_ar_sys_,ppO2_ar_sys_)*self.dcbCO2_dppO2(ppCO2_ar_sys_,ppO2_ar_sys_) ),-1.) * \
            ( self.dcbO2_dppO2(ppCO2_ar_sys_,ppO2_ar_sys_) * (q_vout_l_ * (self.cbCO2(ppCO2_ar_sys_,ppO2_ar_sys_) - self.cbCO2(ppCO2_v_l_,ppO2_v_l_))) - \
                self.dcbCO2_dppO2(ppCO2_ar_sys_,ppO2_ar_sys_) * (q_vout_l_ * (self.cbO2(ppCO2_ar_sys_,ppO2_ar_sys_) - self.cbO2(ppCO2_v_l_,ppO2_v_l_))) )
        # systemic arteries O2
        self.f_[61] = (1./V_ar_sys_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_ar_sys_,ppO2_ar_sys_)*self.dcbO2_dppO2(ppCO2_ar_sys_,ppO2_ar_sys_) - self.dcbO2_dppCO2(ppCO2_ar_sys_,ppO2_ar_sys_)*self.dcbCO2_dppO2(ppCO2_ar_sys_,ppO2_ar_sys_) ),-1.) * \
            ( self.dcbCO2_dppCO2(ppCO2_ar_sys_,ppO2_ar_sys_) * (q_vout_l_ * (self.cbO2(ppCO2_ar_sys_,ppO2_ar_sys_) - self.cbO2(ppCO2_v_l_,ppO2_v_l_))) - \
                self.dcbO2_dppCO2(ppCO2_ar_sys_,ppO2_ar_sys_) * (q_vout_l_ * (self.cbCO2(ppCO2_ar_sys_,ppO2_ar_sys_) - self.cbCO2(ppCO2_v_l_,ppO2_v_l_))) )

        ### systemic capillaries
        # systemic splanchnic arteries CO2
        self.f_[62] = (1./V_arspl_sys_) * sp.Pow(( (self.dcbCO2_dppCO2(ppCO2_arspl_sys_,ppO2_arspl_sys_) + (self.V_tissspl/V_arspl_sys_)*self.dctCO2_dppCO2(ppCO2_arspl_sys_))*(self.dcbO2_dppO2(ppCO2_arspl_sys_,ppO2_arspl_sys_) + (self.V_tissspl/V_arspl_sys_)*self.dctO2_dppO2(ppO2_arspl_sys_)) - self.dcbO2_dppCO2(ppCO2_arspl_sys_,ppO2_arspl_sys_)*self.dcbCO2_dppO2(ppCO2_arspl_sys_,ppO2_arspl_sys_) ),-1.) * \
            ( (self.dcbO2_dppO2(ppCO2_arspl_sys_,ppO2_arspl_sys_) + (self.V_tissspl/V_arspl_sys_)*self.dctO2_dppO2(ppO2_arspl_sys_)) * (q_arspl_sys_in_ * (self.cbCO2(ppCO2_arspl_sys_,ppO2_arspl_sys_) - self.cbCO2(ppCO2_ar_sys_,ppO2_ar_sys_)) - self.M_CO2_arspl) - \
                self.dcbCO2_dppO2(ppCO2_arspl_sys_,ppO2_arspl_sys_) * (q_arspl_sys_in_ * (self.cbO2(ppCO2_arspl_sys_,ppO2_arspl_sys_) - self.cbO2(ppCO2_ar_sys_,ppO2_ar_sys_)) + self.M_O2_arspl*self.ctO2(ppO2_arspl_sys_)/(self.beta_O2+self.ctO2(ppO2_arspl_sys_)) ) )
        # systemic splanchnic arteries O2
        self.f_[63] = (1./V_arspl_sys_) * sp.Pow(( (self.dcbCO2_dppCO2(ppCO2_arspl_sys_,ppO2_arspl_sys_) + (self.V_tissspl/V_arspl_sys_)*self.dctCO2_dppCO2(ppCO2_arspl_sys_))*(self.dcbO2_dppO2(ppCO2_arspl_sys_,ppO2_arspl_sys_) + (self.V_tissspl/V_arspl_sys_)*self.dctO2_dppO2(ppO2_arspl_sys_)) - self.dcbO2_dppCO2(ppCO2_arspl_sys_,ppO2_arspl_sys_)*self.dcbCO2_dppO2(ppCO2_arspl_sys_,ppO2_arspl_sys_) ),-1.) * \
            ( (self.dcbCO2_dppCO2(ppCO2_arspl_sys_,ppO2_arspl_sys_) + (self.V_tissspl/V_arspl_sys_)*self.dctCO2_dppCO2(ppCO2_arspl_sys_)) * (q_arspl_sys_in_ * (self.cbO2(ppCO2_arspl_sys_,ppO2_arspl_sys_) - self.cbO2(ppCO2_ar_sys_,ppO2_ar_sys_)) + self.M_O2_arspl*self.ctO2(ppO2_arspl_sys_)/(self.beta_O2+self.ctO2(ppO2_arspl_sys_))) - \
                self.dcbO2_dppCO2(ppCO2_arspl_sys_,ppO2_arspl_sys_) * (q_arspl_sys_in_ * (self.cbCO2(ppCO2_arspl_sys_,ppO2_arspl_sys_) - self.cbCO2(ppCO2_ar_sys_,ppO2_ar_sys_)) - self.M_CO2_arspl) )

        # systemic extra-esplanchnic arteries CO2
        self.f_[64] = (1./V_arespl_sys_) * sp.Pow(( (self.dcbCO2_dppCO2(ppCO2_arespl_sys_,ppO2_arespl_sys_) + (self.V_tissespl/V_arespl_sys_)*self.dctCO2_dppCO2(ppCO2_arespl_sys_))*(self.dcbO2_dppO2(ppCO2_arespl_sys_,ppO2_arespl_sys_) + (self.V_tissespl/V_arespl_sys_)*self.dctO2_dppO2(ppO2_arespl_sys_)) - self.dcbO2_dppCO2(ppCO2_arespl_sys_,ppO2_arespl_sys_)*self.dcbCO2_dppO2(ppCO2_arespl_sys_,ppO2_arespl_sys_) ),-1.) * \
            ( (self.dcbO2_dppO2(ppCO2_arespl_sys_,ppO2_arespl_sys_) + (self.V_tissespl/V_arespl_sys_)*self.dctO2_dppO2(ppO2_arespl_sys_)) * (q_arespl_sys_in_ * (self.cbCO2(ppCO2_arespl_sys_,ppO2_arespl_sys_) - self.cbCO2(ppCO2_ar_sys_,ppO2_ar_sys_)) - self.M_CO2_arespl) - \
                self.dcbCO2_dppO2(ppCO2_arespl_sys_,ppO2_arespl_sys_) * (q_arespl_sys_in_ * (self.cbO2(ppCO2_arespl_sys_,ppO2_arespl_sys_) - self.cbO2(ppCO2_ar_sys_,ppO2_ar_sys_)) + self.M_O2_arespl*self.ctO2(ppO2_arespl_sys_)/(self.beta_O2+self.ctO2(ppO2_arespl_sys_))) )
        # systemic exrta-splanchnic arteries O2
        self.f_[65] = (1./V_arespl_sys_) * sp.Pow(( (self.dcbCO2_dppCO2(ppCO2_arespl_sys_,ppO2_arespl_sys_) + (self.V_tissespl/V_arespl_sys_)*self.dctCO2_dppCO2(ppCO2_arespl_sys_))*(self.dcbO2_dppO2(ppCO2_arespl_sys_,ppO2_arespl_sys_) + (self.V_tissespl/V_arespl_sys_)*self.dctO2_dppO2(ppO2_arespl_sys_)) - self.dcbO2_dppCO2(ppCO2_arespl_sys_,ppO2_arespl_sys_)*self.dcbCO2_dppO2(ppCO2_arespl_sys_,ppO2_arespl_sys_) ),-1.) * \
            ( (self.dcbCO2_dppCO2(ppCO2_arespl_sys_,ppO2_arespl_sys_) + (self.V_tissespl/V_arespl_sys_)*self.dctCO2_dppCO2(ppCO2_arespl_sys_)) * (q_arespl_sys_in_ * (self.cbO2(ppCO2_arespl_sys_,ppO2_arespl_sys_) - self.cbO2(ppCO2_ar_sys_,ppO2_ar_sys_)) + self.M_O2_arespl*self.ctO2(ppO2_arespl_sys_)/(self.beta_O2+self.ctO2(ppO2_arespl_sys_))) - \
                self.dcbO2_dppCO2(ppCO2_arespl_sys_,ppO2_arespl_sys_) * (q_arespl_sys_in_ * (self.cbCO2(ppCO2_arespl_sys_,ppO2_arespl_sys_) - self.cbCO2(ppCO2_ar_sys_,ppO2_ar_sys_)) - self.M_CO2_arespl) )

        # systemic muscular arteries CO2
        self.f_[66] = (1./V_armsc_sys_) * sp.Pow(( (self.dcbCO2_dppCO2(ppCO2_armsc_sys_,ppO2_armsc_sys_) + (self.V_tissmsc/V_armsc_sys_)*self.dctCO2_dppCO2(ppCO2_armsc_sys_))*(self.dcbO2_dppO2(ppCO2_armsc_sys_,ppO2_armsc_sys_) + (self.V_tissmsc/V_armsc_sys_)*self.dctO2_dppO2(ppO2_armsc_sys_)) - self.dcbO2_dppCO2(ppCO2_armsc_sys_,ppO2_armsc_sys_)*self.dcbCO2_dppO2(ppCO2_armsc_sys_,ppO2_armsc_sys_) ),-1.) * \
            ( (self.dcbO2_dppO2(ppCO2_armsc_sys_,ppO2_armsc_sys_) + (self.V_tissmsc/V_armsc_sys_)*self.dctO2_dppO2(ppO2_armsc_sys_)) * (q_armsc_sys_in_ * (self.cbCO2(ppCO2_armsc_sys_,ppO2_armsc_sys_) - self.cbCO2(ppCO2_ar_sys_,ppO2_ar_sys_)) - self.M_CO2_armsc) - \
                self.dcbCO2_dppO2(ppCO2_armsc_sys_,ppO2_armsc_sys_) * (q_armsc_sys_in_ * (self.cbO2(ppCO2_armsc_sys_,ppO2_armsc_sys_) - self.cbO2(ppCO2_ar_sys_,ppO2_ar_sys_)) + self.M_O2_armsc*self.ctO2(ppO2_armsc_sys_)/(self.beta_O2+self.ctO2(ppO2_armsc_sys_))) )
        # systemic muscular arteries O2
        self.f_[67] = (1./V_armsc_sys_) * sp.Pow(( (self.dcbCO2_dppCO2(ppCO2_armsc_sys_,ppO2_armsc_sys_) + (self.V_tissmsc/V_armsc_sys_)*self.dctCO2_dppCO2(ppCO2_armsc_sys_))*(self.dcbO2_dppO2(ppCO2_armsc_sys_,ppO2_armsc_sys_) + (self.V_tissmsc/V_armsc_sys_)*self.dctO2_dppO2(ppO2_armsc_sys_)) - self.dcbO2_dppCO2(ppCO2_armsc_sys_,ppO2_armsc_sys_)*self.dcbCO2_dppO2(ppCO2_armsc_sys_,ppO2_armsc_sys_) ),-1.) * \
            ( (self.dcbCO2_dppCO2(ppCO2_armsc_sys_,ppO2_armsc_sys_) + (self.V_tissmsc/V_armsc_sys_)*self.dctCO2_dppCO2(ppCO2_armsc_sys_)) * (q_armsc_sys_in_ * (self.cbO2(ppCO2_armsc_sys_,ppO2_armsc_sys_) - self.cbO2(ppCO2_ar_sys_,ppO2_ar_sys_)) + self.M_O2_armsc*self.ctO2(ppO2_armsc_sys_)/(self.beta_O2+self.ctO2(ppO2_armsc_sys_))) - \
                self.dcbO2_dppCO2(ppCO2_armsc_sys_,ppO2_armsc_sys_) * (q_armsc_sys_in_ * (self.cbCO2(ppCO2_armsc_sys_,ppO2_armsc_sys_) - self.cbCO2(ppCO2_ar_sys_,ppO2_ar_sys_)) - self.M_CO2_armsc) )

        # systemic cerebral arteries CO2
        self.f_[68] = (1./V_arcer_sys_) * sp.Pow(( (self.dcbCO2_dppCO2(ppCO2_arcer_sys_,ppO2_arcer_sys_) + (self.V_tisscer/V_arcer_sys_)*self.dctCO2_dppCO2(ppCO2_arcer_sys_))*(self.dcbO2_dppO2(ppCO2_arcer_sys_,ppO2_arcer_sys_) + (self.V_tisscer/V_arcer_sys_)*self.dctO2_dppO2(ppO2_arcer_sys_)) - self.dcbO2_dppCO2(ppCO2_arcer_sys_,ppO2_arcer_sys_)*self.dcbCO2_dppO2(ppCO2_arcer_sys_,ppO2_arcer_sys_) ),-1.) * \
            ( (self.dcbO2_dppO2(ppCO2_arcer_sys_,ppO2_arcer_sys_) + (self.V_tisscer/V_arcer_sys_)*self.dctO2_dppO2(ppO2_arcer_sys_)) * (q_arcer_sys_in_ * (self.cbCO2(ppCO2_arcer_sys_,ppO2_arcer_sys_) - self.cbCO2(ppCO2_ar_sys_,ppO2_ar_sys_)) - self.M_CO2_arcer) - \
                self.dcbCO2_dppO2(ppCO2_arcer_sys_,ppO2_arcer_sys_) * (q_arcer_sys_in_ * (self.cbO2(ppCO2_arcer_sys_,ppO2_arcer_sys_) - self.cbO2(ppCO2_ar_sys_,ppO2_ar_sys_)) + self.M_O2_arcer*self.ctO2(ppO2_arcer_sys_)/(self.beta_O2+self.ctO2(ppO2_arcer_sys_))) )
        # systemic cerebral arteries O2
        self.f_[69] = (1./V_arcer_sys_) * sp.Pow(( (self.dcbCO2_dppCO2(ppCO2_arcer_sys_,ppO2_arcer_sys_) + (self.V_tisscer/V_arcer_sys_)*self.dctCO2_dppCO2(ppCO2_arcer_sys_))*(self.dcbO2_dppO2(ppCO2_arcer_sys_,ppO2_arcer_sys_) + (self.V_tisscer/V_arcer_sys_)*self.dctO2_dppO2(ppO2_arcer_sys_)) - self.dcbO2_dppCO2(ppCO2_arcer_sys_,ppO2_arcer_sys_)*self.dcbCO2_dppO2(ppCO2_arcer_sys_,ppO2_arcer_sys_) ),-1.) * \
            ( (self.dcbCO2_dppCO2(ppCO2_arcer_sys_,ppO2_arcer_sys_) + (self.V_tisscer/V_arcer_sys_)*self.dctCO2_dppCO2(ppCO2_arcer_sys_)) * (q_arcer_sys_in_ * (self.cbO2(ppCO2_arcer_sys_,ppO2_arcer_sys_) - self.cbO2(ppCO2_ar_sys_,ppO2_ar_sys_)) + self.M_O2_arcer*self.ctO2(ppO2_arcer_sys_)/(self.beta_O2+self.ctO2(ppO2_arcer_sys_))) - \
                self.dcbO2_dppCO2(ppCO2_arcer_sys_,ppO2_arcer_sys_) * (q_arcer_sys_in_ * (self.cbCO2(ppCO2_arcer_sys_,ppO2_arcer_sys_) - self.cbCO2(ppCO2_ar_sys_,ppO2_ar_sys_)) - self.M_CO2_arcer) )

        # systemic coronary arteries CO2
        self.f_[70] = (1./V_arcor_sys_) * sp.Pow(( (self.dcbCO2_dppCO2(ppCO2_arcor_sys_,ppO2_arcor_sys_) + (self.V_tisscor/V_arcor_sys_)*self.dctCO2_dppCO2(ppCO2_arcor_sys_))*(self.dcbO2_dppO2(ppCO2_arcor_sys_,ppO2_arcor_sys_) + (self.V_tisscor/V_arcor_sys_)*self.dctO2_dppO2(ppO2_arcor_sys_)) - self.dcbO2_dppCO2(ppCO2_arcor_sys_,ppO2_arcor_sys_)*self.dcbCO2_dppO2(ppCO2_arcor_sys_,ppO2_arcor_sys_) ),-1.) * \
            ( (self.dcbO2_dppO2(ppCO2_arcor_sys_,ppO2_arcor_sys_) + (self.V_tisscor/V_arcor_sys_)*self.dctO2_dppO2(ppO2_arcor_sys_)) * (q_arcor_sys_in_ * (self.cbCO2(ppCO2_arcor_sys_,ppO2_arcor_sys_) - self.cbCO2(ppCO2_ar_sys_,ppO2_ar_sys_)) - self.M_CO2_arcor) - \
                self.dcbCO2_dppO2(ppCO2_arcor_sys_,ppO2_arcor_sys_) * (q_arcor_sys_in_ * (self.cbO2(ppCO2_arcor_sys_,ppO2_arcor_sys_) - self.cbO2(ppCO2_ar_sys_,ppO2_ar_sys_)) + self.M_O2_arcor*self.ctO2(ppO2_arcor_sys_)/(self.beta_O2+self.ctO2(ppO2_arcor_sys_))) )
        # systemic coronary arteries O2
        self.f_[71] = (1./V_arcor_sys_) * sp.Pow(( (self.dcbCO2_dppCO2(ppCO2_arcor_sys_,ppO2_arcor_sys_) + (self.V_tisscor/V_arcor_sys_)*self.dctCO2_dppCO2(ppCO2_arcor_sys_))*(self.dcbO2_dppO2(ppCO2_arcor_sys_,ppO2_arcor_sys_) + (self.V_tisscor/V_arcor_sys_)*self.dctO2_dppO2(ppO2_arcor_sys_)) - self.dcbO2_dppCO2(ppCO2_arcor_sys_,ppO2_arcor_sys_)*self.dcbCO2_dppO2(ppCO2_arcor_sys_,ppO2_arcor_sys_) ),-1.) * \
            ( (self.dcbCO2_dppCO2(ppCO2_arcor_sys_,ppO2_arcor_sys_) + (self.V_tisscor/V_arcor_sys_)*self.dctCO2_dppCO2(ppCO2_arcor_sys_)) * (q_arcor_sys_in_ * (self.cbO2(ppCO2_arcor_sys_,ppO2_arcor_sys_) - self.cbO2(ppCO2_ar_sys_,ppO2_ar_sys_)) + self.M_O2_arcor*self.ctO2(ppO2_arcor_sys_)/(self.beta_O2+self.ctO2(ppO2_arcor_sys_))) - \
                self.dcbO2_dppCO2(ppCO2_arcor_sys_,ppO2_arcor_sys_) * (q_arcor_sys_in_ * (self.cbCO2(ppCO2_arcor_sys_,ppO2_arcor_sys_) - self.cbCO2(ppCO2_ar_sys_,ppO2_ar_sys_)) - self.M_CO2_arcor) )

        # systemic splanchnic veins CO2
        self.f_[72] = (1./V_venspl_sys_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_venspl_sys_,ppO2_venspl_sys_)*self.dcbO2_dppO2(ppCO2_venspl_sys_,ppO2_venspl_sys_) - self.dcbO2_dppCO2(ppCO2_venspl_sys_,ppO2_venspl_sys_)*self.dcbCO2_dppO2(ppCO2_venspl_sys_,ppO2_venspl_sys_) ),-1.) * \
            ( self.dcbO2_dppO2(ppCO2_venspl_sys_,ppO2_venspl_sys_) * (q_arspl_sys_ * (self.cbCO2(ppCO2_venspl_sys_,ppO2_venspl_sys_) - self.cbCO2(ppCO2_arspl_sys_,ppO2_arspl_sys_))) - \
                self.dcbCO2_dppO2(ppCO2_venspl_sys_,ppO2_venspl_sys_) * (q_arspl_sys_ * (self.cbO2(ppCO2_venspl_sys_,ppO2_venspl_sys_) - self.cbO2(ppCO2_arspl_sys_,ppO2_arspl_sys_))))
        # systemic splanchnic veins O2
        self.f_[73]= (1./V_venspl_sys_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_venspl_sys_,ppO2_venspl_sys_)*self.dcbO2_dppO2(ppCO2_venspl_sys_,ppO2_venspl_sys_) - self.dcbO2_dppCO2(ppCO2_venspl_sys_,ppO2_venspl_sys_)*self.dcbCO2_dppO2(ppCO2_venspl_sys_,ppO2_venspl_sys_) ),-1.) * \
            ( self.dcbCO2_dppCO2(ppCO2_venspl_sys_,ppO2_venspl_sys_) * (q_arspl_sys_ * (self.cbO2(ppCO2_venspl_sys_,ppO2_venspl_sys_) - self.cbO2(ppCO2_arspl_sys_,ppO2_arspl_sys_))) - \
                self.dcbO2_dppCO2(ppCO2_venspl_sys_,ppO2_venspl_sys_) * (q_arspl_sys_ * (self.cbCO2(ppCO2_venspl_sys_,ppO2_venspl_sys_) - self.cbCO2(ppCO2_arspl_sys_,ppO2_arspl_sys_))) )

        # systemic extra-splanchnic veins CO2
        self.f_[74] = (1./V_venespl_sys_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_venespl_sys_,ppO2_venespl_sys_)*self.dcbO2_dppO2(ppCO2_venespl_sys_,ppO2_venespl_sys_) - self.dcbO2_dppCO2(ppCO2_venespl_sys_,ppO2_venespl_sys_)*self.dcbCO2_dppO2(ppCO2_venespl_sys_,ppO2_venespl_sys_) ),-1.) * \
            ( self.dcbO2_dppO2(ppCO2_venespl_sys_,ppO2_venespl_sys_) * (q_arespl_sys_ * (self.cbCO2(ppCO2_venespl_sys_,ppO2_venespl_sys_) - self.cbCO2(ppCO2_arespl_sys_,ppO2_arespl_sys_))) - \
                self.dcbCO2_dppO2(ppCO2_venespl_sys_,ppO2_venespl_sys_) * (q_arespl_sys_ * (self.cbO2(ppCO2_venespl_sys_,ppO2_venespl_sys_) - self.cbO2(ppCO2_arespl_sys_,ppO2_arespl_sys_))))
        # systemic extra-splanchnic veins O2
        self.f_[75] = (1./V_venespl_sys_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_venespl_sys_,ppO2_venespl_sys_)*self.dcbO2_dppO2(ppCO2_venespl_sys_,ppO2_venespl_sys_) - self.dcbO2_dppCO2(ppCO2_venespl_sys_,ppO2_venespl_sys_)*self.dcbCO2_dppO2(ppCO2_venespl_sys_,ppO2_venespl_sys_) ),-1.) * \
            ( self.dcbCO2_dppCO2(ppCO2_venespl_sys_,ppO2_venespl_sys_) * (q_arespl_sys_ * (self.cbO2(ppCO2_venespl_sys_,ppO2_venespl_sys_) - self.cbO2(ppCO2_arespl_sys_,ppO2_arespl_sys_))) - \
                self.dcbO2_dppCO2(ppCO2_venespl_sys_,ppO2_venespl_sys_) * (q_arespl_sys_ * (self.cbCO2(ppCO2_venespl_sys_,ppO2_venespl_sys_) - self.cbCO2(ppCO2_arespl_sys_,ppO2_arespl_sys_))) )

        # systemic muscular veins CO2
        self.f_[76] = (1./V_venmsc_sys_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_venmsc_sys_,ppO2_venmsc_sys_)*self.dcbO2_dppO2(ppCO2_venmsc_sys_,ppO2_venmsc_sys_) - self.dcbO2_dppCO2(ppCO2_venmsc_sys_,ppO2_venmsc_sys_)*self.dcbCO2_dppO2(ppCO2_venmsc_sys_,ppO2_venmsc_sys_) ),-1.) * \
            ( self.dcbO2_dppO2(ppCO2_venmsc_sys_,ppO2_venmsc_sys_) * (q_armsc_sys_ * (self.cbCO2(ppCO2_venmsc_sys_,ppO2_venmsc_sys_) - self.cbCO2(ppCO2_armsc_sys_,ppO2_armsc_sys_))) - \
                self.dcbCO2_dppO2(ppCO2_venmsc_sys_,ppO2_venmsc_sys_) * (q_armsc_sys_ * (self.cbO2(ppCO2_venmsc_sys_,ppO2_venmsc_sys_) - self.cbO2(ppCO2_armsc_sys_,ppO2_armsc_sys_))))
        # systemic muscular veins O2
        self.f_[77] = (1./V_venmsc_sys_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_venmsc_sys_,ppO2_venmsc_sys_)*self.dcbO2_dppO2(ppCO2_venmsc_sys_,ppO2_venmsc_sys_) - self.dcbO2_dppCO2(ppCO2_venmsc_sys_,ppO2_venmsc_sys_)*self.dcbCO2_dppO2(ppCO2_venmsc_sys_,ppO2_venmsc_sys_) ),-1.) * \
            ( self.dcbCO2_dppCO2(ppCO2_venmsc_sys_,ppO2_venmsc_sys_) * (q_armsc_sys_ * (self.cbO2(ppCO2_venmsc_sys_,ppO2_venmsc_sys_) - self.cbO2(ppCO2_armsc_sys_,ppO2_armsc_sys_))) - \
                self.dcbO2_dppCO2(ppCO2_venmsc_sys_,ppO2_venmsc_sys_) * (q_armsc_sys_ * (self.cbCO2(ppCO2_venmsc_sys_,ppO2_venmsc_sys_) - self.cbCO2(ppCO2_armsc_sys_,ppO2_armsc_sys_))) )

        # systemic cerebral veins CO2
        self.f_[78] = (1./V_vencer_sys_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_vencer_sys_,ppO2_vencer_sys_)*self.dcbO2_dppO2(ppCO2_vencer_sys_,ppO2_vencer_sys_) - self.dcbO2_dppCO2(ppCO2_vencer_sys_,ppO2_vencer_sys_)*self.dcbCO2_dppO2(ppCO2_vencer_sys_,ppO2_vencer_sys_) ),-1.) * \
            ( self.dcbO2_dppO2(ppCO2_vencer_sys_,ppO2_vencer_sys_) * (q_arcer_sys_ * (self.cbCO2(ppCO2_vencer_sys_,ppO2_vencer_sys_) - self.cbCO2(ppCO2_arcer_sys_,ppO2_arcer_sys_))) - \
                self.dcbCO2_dppO2(ppCO2_vencer_sys_,ppO2_vencer_sys_) * (q_arcer_sys_ * (self.cbO2(ppCO2_vencer_sys_,ppO2_vencer_sys_) - self.cbO2(ppCO2_arcer_sys_,ppO2_arcer_sys_))))
        # systemic cerebral veins O2
        self.f_[79] = (1./V_vencer_sys_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_vencer_sys_,ppO2_vencer_sys_)*self.dcbO2_dppO2(ppCO2_vencer_sys_,ppO2_vencer_sys_) - self.dcbO2_dppCO2(ppCO2_vencer_sys_,ppO2_vencer_sys_)*self.dcbCO2_dppO2(ppCO2_vencer_sys_,ppO2_vencer_sys_) ),-1.) * \
            ( self.dcbCO2_dppCO2(ppCO2_vencer_sys_,ppO2_vencer_sys_) * (q_arcer_sys_ * (self.cbO2(ppCO2_vencer_sys_,ppO2_vencer_sys_) - self.cbO2(ppCO2_arcer_sys_,ppO2_arcer_sys_))) - \
                self.dcbO2_dppCO2(ppCO2_vencer_sys_,ppO2_vencer_sys_) * (q_arcer_sys_ * (self.cbCO2(ppCO2_vencer_sys_,ppO2_vencer_sys_) - self.cbCO2(ppCO2_arcer_sys_,ppO2_arcer_sys_))) )

        # systemic coronary veins CO2
        self.f_[80] = (1./V_vencor_sys_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_vencor_sys_,ppO2_vencor_sys_)*self.dcbO2_dppO2(ppCO2_vencor_sys_,ppO2_vencor_sys_) - self.dcbO2_dppCO2(ppCO2_vencor_sys_,ppO2_vencor_sys_)*self.dcbCO2_dppO2(ppCO2_vencor_sys_,ppO2_vencor_sys_) ),-1.) * \
            ( self.dcbO2_dppO2(ppCO2_vencor_sys_,ppO2_vencor_sys_) * (q_arcor_sys_ * (self.cbCO2(ppCO2_vencor_sys_,ppO2_vencor_sys_) - self.cbCO2(ppCO2_arcor_sys_,ppO2_arcor_sys_))) - \
                self.dcbCO2_dppO2(ppCO2_vencor_sys_,ppO2_vencor_sys_) * (q_arcor_sys_ * (self.cbO2(ppCO2_vencor_sys_,ppO2_vencor_sys_) - self.cbO2(ppCO2_arcor_sys_,ppO2_arcor_sys_))))
        # systemic coronary veins O2
        self.f_[81] = (1./V_vencor_sys_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_vencor_sys_,ppO2_vencor_sys_)*self.dcbO2_dppO2(ppCO2_vencor_sys_,ppO2_vencor_sys_) - self.dcbO2_dppCO2(ppCO2_vencor_sys_,ppO2_vencor_sys_)*self.dcbCO2_dppO2(ppCO2_vencor_sys_,ppO2_vencor_sys_) ),-1.) * \
            ( self.dcbCO2_dppCO2(ppCO2_vencor_sys_,ppO2_vencor_sys_) * (q_arcor_sys_ * (self.cbO2(ppCO2_vencor_sys_,ppO2_vencor_sys_) - self.cbO2(ppCO2_arcor_sys_,ppO2_arcor_sys_))) - \
                self.dcbO2_dppCO2(ppCO2_vencor_sys_,ppO2_vencor_sys_) * (q_arcor_sys_ * (self.cbCO2(ppCO2_vencor_sys_,ppO2_vencor_sys_) - self.cbCO2(ppCO2_arcor_sys_,ppO2_arcor_sys_))) )

        # mixture rule for joining flows: c_upstr = (q_upstr_1 * c_upstr_1 + ... + q_upstr_n * c_upstr_n) / (q_upstr_1 + ... + q_upstr_n)
        # systemic veins CO2
        self.f_[82] = (1./V_ven_sys_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_ven_sys_,ppO2_ven_sys_)*self.dcbO2_dppO2(ppCO2_ven_sys_,ppO2_ven_sys_) - self.dcbO2_dppCO2(ppCO2_ven_sys_,ppO2_ven_sys_)*self.dcbCO2_dppO2(ppCO2_ven_sys_,ppO2_ven_sys_) ),-1.) * \
            ( self.dcbO2_dppO2(ppCO2_ven_sys_,ppO2_ven_sys_) * ( ((q_venspl_sys_+q_venespl_sys_+q_venmsc_sys_+q_vencer_sys_+q_vencor_sys_)*self.cbCO2(ppCO2_ven_sys_,ppO2_ven_sys_) - (q_venspl_sys_*self.cbCO2(ppCO2_venspl_sys_,ppO2_venspl_sys_) + q_venespl_sys_*self.cbCO2(ppCO2_venespl_sys_,ppO2_venespl_sys_) + q_venmsc_sys_*self.cbCO2(ppCO2_venmsc_sys_,ppO2_venmsc_sys_) + q_vencer_sys_*self.cbCO2(ppCO2_vencer_sys_,ppO2_vencer_sys_) + q_vencor_sys_*self.cbCO2(ppCO2_vencor_sys_,ppO2_vencor_sys_) ))) - \
                self.dcbCO2_dppO2(ppCO2_ven_sys_,ppO2_ven_sys_) * ( ((q_venspl_sys_+q_venespl_sys_+q_venmsc_sys_+q_vencer_sys_+q_vencor_sys_)*self.cbO2(ppCO2_ven_sys_,ppO2_ven_sys_) - (q_venspl_sys_*self.cbO2(ppCO2_venspl_sys_,ppO2_venspl_sys_) + q_venespl_sys_*self.cbO2(ppCO2_venespl_sys_,ppO2_venespl_sys_) + q_venmsc_sys_*self.cbO2(ppCO2_venmsc_sys_,ppO2_venmsc_sys_) + q_vencer_sys_*self.cbO2(ppCO2_vencer_sys_,ppO2_vencer_sys_) + q_vencor_sys_*self.cbO2(ppCO2_vencor_sys_,ppO2_vencor_sys_) ))))
        # systemic veins O2
        self.f_[83] = (1./V_ven_sys_) * sp.Pow(( self.dcbCO2_dppCO2(ppCO2_ven_sys_,ppO2_ven_sys_)*self.dcbO2_dppO2(ppCO2_ven_sys_,ppO2_ven_sys_) - self.dcbO2_dppCO2(ppCO2_ven_sys_,ppO2_ven_sys_)*self.dcbCO2_dppO2(ppCO2_ven_sys_,ppO2_ven_sys_) ),-1.) * \
            ( self.dcbCO2_dppCO2(ppCO2_ven_sys_,ppO2_ven_sys_) * ( ((q_venspl_sys_+q_venespl_sys_+q_venmsc_sys_+q_vencer_sys_+q_vencor_sys_)*self.cbO2(ppCO2_ven_sys_,ppO2_ven_sys_) - (q_venspl_sys_*self.cbO2(ppCO2_venspl_sys_,ppO2_venspl_sys_) + q_venespl_sys_*self.cbO2(ppCO2_venespl_sys_,ppO2_venespl_sys_) + q_venmsc_sys_*self.cbO2(ppCO2_venmsc_sys_,ppO2_venmsc_sys_) + q_vencer_sys_*self.cbO2(ppCO2_vencer_sys_,ppO2_vencer_sys_) + q_vencor_sys_*self.cbO2(ppCO2_vencor_sys_,ppO2_vencor_sys_) ))) - \
                self.dcbO2_dppCO2(ppCO2_ven_sys_,ppO2_ven_sys_) * ( ((q_venspl_sys_+q_venespl_sys_+q_venmsc_sys_+q_vencer_sys_+q_vencor_sys_)*self.cbCO2(ppCO2_ven_sys_,ppO2_ven_sys_) - (q_venspl_sys_*self.cbCO2(ppCO2_venspl_sys_,ppO2_venspl_sys_) + q_venespl_sys_*self.cbCO2(ppCO2_venespl_sys_,ppO2_venespl_sys_) + q_venmsc_sys_*self.cbCO2(ppCO2_venmsc_sys_,ppO2_venmsc_sys_) + q_vencer_sys_*self.cbCO2(ppCO2_vencer_sys_,ppO2_vencer_sys_) + q_vencor_sys_*self.cbCO2(ppCO2_vencor_sys_,ppO2_vencor_sys_) ))) )

        # add to auxiliary variable vector (mainly in order to store quantities for post-processing)
        self.a_[51] = self.SO2(ppCO2_ar_pul_,ppO2_ar_pul_)
        self.a_[61] = self.SO2(ppCO2_ar_sys_,ppO2_ar_sys_)


    def initialize(self, var, iniparam):

        cardiovascular0Dsyspulcap.initialize(self, var, iniparam)

        # initial value of time-varying pleural pressure
        U_t_0 = self.U_t()

        V_alv_0 = iniparam['V_alv_0']
        if V_alv_0>=0: var[36] = V_alv_0
        if V_alv_0<0: var[36] = (self.U_m - U_t_0)/self.E_alv + self.V_lung_u

        var[37] = iniparam['q_alv_0']

        p_alv_0 = iniparam['p_alv_0']
        if p_alv_0>=0: var[38] = p_alv_0
        if p_alv_0<0: var[38] = self.U_m

        var[39] = iniparam['fCO2_alv_0']
        var[40] = iniparam['fO2_alv_0']
        var[41] = iniparam['q_arspl_sys_in_0']
        var[42] = iniparam['q_arespl_sys_in_0']
        var[43] = iniparam['q_armsc_sys_in_0']
        var[44] = iniparam['q_arcer_sys_in_0']
        var[45] = iniparam['q_arcor_sys_in_0']
        var[46] = iniparam['ppCO2_at_r_0']
        var[47] = iniparam['ppO2_at_r_0']
        var[48] = iniparam['ppCO2_v_r_0']
        var[49] = iniparam['ppO2_v_r_0']
        var[50] = iniparam['ppCO2_ar_pul_0']
        var[51] = iniparam['ppO2_ar_pul_0']
        var[52] = iniparam['ppCO2_cap_pul_0']
        var[53] = iniparam['ppO2_cap_pul_0']
        var[54] = iniparam['ppCO2_ven_pul_0']
        var[55] = iniparam['ppO2_ven_pul_0']
        var[56] = iniparam['ppCO2_at_l_0']
        var[57] = iniparam['ppO2_at_l_0']
        var[58] = iniparam['ppCO2_v_l_0']
        var[59] = iniparam['ppO2_v_l_0']
        var[60] = iniparam['ppCO2_ar_sys_0']
        var[61] = iniparam['ppO2_ar_sys_0']
        var[62] = iniparam['ppCO2_arspl_sys_0']
        var[63] = iniparam['ppO2_arspl_sys_0']
        var[64] = iniparam['ppCO2_arespl_sys_0']
        var[65] = iniparam['ppO2_arespl_sys_0']
        var[66] = iniparam['ppCO2_armsc_sys_0']
        var[67] = iniparam['ppO2_armsc_sys_0']
        var[68] = iniparam['ppCO2_arcer_sys_0']
        var[69] = iniparam['ppO2_arcer_sys_0']
        var[70] = iniparam['ppCO2_arcor_sys_0']
        var[71] = iniparam['ppO2_arcor_sys_0']
        var[72] = iniparam['ppCO2_venspl_sys_0']
        var[73] = iniparam['ppO2_venspl_sys_0']
        var[74] = iniparam['ppCO2_venespl_sys_0']
        var[75] = iniparam['ppO2_venespl_sys_0']
        var[76] = iniparam['ppCO2_venmsc_sys_0']
        var[77] = iniparam['ppO2_venmsc_sys_0']
        var[78] = iniparam['ppCO2_vencer_sys_0']
        var[79] = iniparam['ppO2_vencer_sys_0']
        var[80] = iniparam['ppCO2_vencor_sys_0']
        var[81] = iniparam['ppO2_vencor_sys_0']
        var[82] = iniparam['ppCO2_ven_sys_0']
        var[83] = iniparam['ppO2_ven_sys_0']


    # time-varying pleural pressure
    def U_t(self):

        # Ben-Tal, J Theor Biol (2006) p. 491
        return self.U_m - self.R_airw*self.omega_breath * (self.V_lung_tidal/2.)*sp.sin(self.omega_breath*self.t_) - \
            self.E_alv*(self.V_lung_total-(self.V_lung_tidal/2.)*sp.cos(self.omega_breath*self.t_))


    # cbO2 and its derivatives
    def cbO2(self, ppCO2, ppO2):

        # with Hill oxygen dissociation curve - simplest form, independent of CO2 and pH !
        cbO2_val = self.alpha_O2 * ppO2 + self.c_Hb * self.SO2(ppCO2,ppO2)

        return cbO2_val

    def SO2(self, ppCO2, ppO2):

        n = 2.7
        ppO2_50 = 26.8/7.500615 # 26.8 mmHg -> convert to kPa!
        # with Hill oxygen dissociation curve - simplest form, independent of CO2 and pH !

        SO2_val = sp.Pow((ppO2/ppO2_50),n) / (1. + sp.Pow((ppO2/ppO2_50),n))

        return SO2_val

    # w.r.t. O2
    def dcbO2_dppO2(self, ppCO2, ppO2):

        n = 2.7
        ppO2_50 = 26.8/7.500615 # 26.8 mmHg -> convert to kPa!

        dcbO2_dppO2_val = self.alpha_O2 + self.c_Hb * n * sp.Pow((ppO2/ppO2_50),n)/(sp.Pow((1.+sp.Pow((ppO2/ppO2_50),n)),2.) * ppO2)

        return dcbO2_dppO2_val


    # w.r.t. CO2
    def dcbO2_dppCO2(self, ppCO2, ppO2):

        dcbO2_dppCO2_val = 0.

        return dcbO2_dppCO2_val


    # cbCO2 and its derivatives
    def cbCO2(self, ppCO2, ppO2):

        cbCO2_val = self.alpha_CO2 * ppCO2

        return cbCO2_val


    # w.r.t. CO2
    def dcbCO2_dppCO2(self, ppCO2, ppO2):

        dcbCO2_dppCO2_val = self.alpha_CO2

        return dcbCO2_dppCO2_val


    # w.r.t. O2
    def dcbCO2_dppO2(self, ppCO2, ppO2):

        dcbCO2_dppO2_val = 0.

        return dcbCO2_dppO2_val


    def ctO2(self, ppO2):

        ctO2_val = self.alpha_O2 * ppO2

        return ctO2_val


    def dctO2_dppO2(self, ppO2):

        dctO2_dppO2_val = self.alpha_O2

        return dctO2_dppO2_val


    def ctCO2(self, ppCO2):

        ctCO2_val = self.alpha_CO2 * ppCO2

        return ctCO2_val


    def dctCO2_dppCO2(self, ppCO2):

        dctCO2_dppCO2_val = self.alpha_CO2

        return dctCO2_dppCO2_val


    def check_periodic(self, varTc, varTc_old, auxTc, auxTc_old, eps, check, cyclerr):

        if isinstance(varTc, np.ndarray): varTc_sq, varTc_old_sq = varTc, varTc_old
        else: varTc_sq, varTc_old_sq = allgather_vec(varTc, self.comm), allgather_vec(varTc_old, self.comm)

        vals = []

        # could get critical here since the respiratory cycle may differ from the heart cycle! So the oscillatory lung dofs should be excluded
        oscillatory_lung_dofs=[36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55]

        if check[0]=='allvar':

            var_ids, aux_ids = list(self.varmap.values()), []

        elif check[0]=='allvaraux':

            var_ids, aux_ids = list(self.varmap.values()), list(self.auxmap.values())

        elif check[0]=='pvar':

            var_ids, aux_ids = [1,3,4,6,8,14,16,18,20,22,24,27,29,30,32,34,
                        46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83], []

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
            if i in var_ids and i not in oscillatory_lung_dofs:
                vals.append( math.fabs((varTc_sq[i]-varTc_old_sq[i])/max(1.0,math.fabs(varTc_old_sq[i]))) )

        for i in range(len(auxTc)):
            if i in aux_ids:
                vals.append( math.fabs((auxTc[i]-auxTc_old[i])/max(1.0,math.fabs(auxTc_old[i]))) )

        cyclerr[0] = max(vals)

        if cyclerr[0] <= eps:
            is_periodic = True
        else:
            is_periodic = False

        return is_periodic


    def print_to_screen(self, var, aux):

        cardiovascular0Dsyspulcap.print_to_screen(self, var, aux)

        if isinstance(var, np.ndarray): var_sq = var
        else: var_sq = allgather_vec(var, self.comm)

        if self.comm.rank == 0:

            print("Output of 0D respiratory model (syspulcaprespir):")

            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<12s}{:<3s}{:<10.3f}'.format('SO2_ar_sys',' = ',aux[self.auxmap['SO2_ar_sys']],'   ','SO2_ar_pul',' = ',aux[self.auxmap['SO2_ar_pul']]))

            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<12s}{:<3s}{:<10.3f}'.format('ppO2_ar_sys',' = ',var_sq[self.varmap['ppO2_ar_sys']],'   ','ppO2_ar_pul',' = ',var_sq[self.varmap['ppO2_ar_pul']]))
            print('{:<12s}{:<3s}{:<10.3f}{:<3s}{:<12s}{:<3s}{:<10.3f}'.format('ppCO2_ar_sys',' = ',var_sq[self.varmap['ppCO2_ar_sys']],'   ','ppCO2_ar_pul',' = ',var_sq[self.varmap['ppCO2_ar_pul']]))

            sys.stdout.flush()

        if not isinstance(var, np.ndarray): del var_sq




def postprocess_groups_syspulcaprespir(groups, coronarymodel=None, indpertaftercyl=0,multiscalegandr=False):

    import cardiovascular0D_syspulcap

    cardiovascular0D_syspulcap.postprocess_groups_syspulcap(groups,indpertaftercyl,multiscalegandr)

    # index 14
    groups.append({'ppO2_time_sys_l'  : ['ppO2_at_l', 'ppO2_v_l', 'ppO2_ar_sys', 'ppO2_arspl_sys', 'ppO2_arespl_sys', 'ppO2_armsc_sys', 'ppO2_arcer_sys', 'ppO2_arcor_sys', 'ppO2_venspl_sys', 'ppO2_venespl_sys', 'ppO2_venmsc_sys', 'ppO2_vencer_sys', 'ppO2_vencor_sys', 'ppO2_ven_sys'],
                   'tex'              : ['$p_{\\\mathrm{O}_2,\\\mathrm{at}}^{\\\ell}$', '$p_{\\\mathrm{O}_2,\\\mathrm{v}}^{\\\ell}$', '$p_{\\\mathrm{O}_2,\\\mathrm{ar}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{O}_2,\\\mathrm{ar,spl}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{O}_2,\\\mathrm{ar,espl}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{O}_2,\\\mathrm{ar,msc}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{O}_2,\\\mathrm{ar,cer}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{O}_2,\\\mathrm{ar,cor}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{O}_2,\\\mathrm{ven,spl}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{O}_2,\\\mathrm{ven,espl}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{O}_2,\\\mathrm{ven,msc}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{O}_2,\\\mathrm{ven,cer}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{O}_2,\\\mathrm{ven,cor}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{O}_2,\\\mathrm{ven}}^{\\\mathrm{sys}}$'],
                   'lines'            : [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]})
    # index 15
    groups.append({'ppCO2_time_sys_l' : ['ppCO2_at_l', 'ppCO2_v_l', 'ppCO2_ar_sys', 'ppCO2_arspl_sys', 'ppCO2_arespl_sys', 'ppCO2_armsc_sys', 'ppCO2_arcer_sys', 'ppCO2_arcor_sys', 'ppCO2_venspl_sys', 'ppCO2_venespl_sys', 'ppCO2_venmsc_sys', 'ppCO2_vencer_sys', 'ppCO2_vencor_sys', 'ppCO2_ven_sys'],
                   'tex'              : ['$p_{\\\mathrm{CO}_2,\\\mathrm{at}}^{\\\ell}$', '$p_{\\\mathrm{CO}_2,\\\mathrm{v}}^{\\\ell}$', '$p_{\\\mathrm{CO}_2,\\\mathrm{ar}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{CO}_2,\\\mathrm{ar,spl}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{CO}_2,\\\mathrm{ar,espl}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{CO}_2,\\\mathrm{ar,msc}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{CO}_2,\\\mathrm{ar,cer}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{CO}_2,\\\mathrm{ar,cor}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{CO}_2,\\\mathrm{ven,spl}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{CO}_2,\\\mathrm{ven,espl}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{CO}_2,\\\mathrm{ven,msc}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{CO}_2,\\\mathrm{ven,cer}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{CO}_2,\\\mathrm{ven,cor}}^{\\\mathrm{sys}}$', '$p_{\\\mathrm{CO}_2,\\\mathrm{ven}}^{\\\mathrm{sys}}$'],
                   'lines'            : [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]})
    # index 16
    groups.append({'ppO2_time_pul_r'  : ['ppO2_at_r', 'ppO2_v_r', 'ppO2_ar_pul', 'ppO2_ven_pul', 'ppO2_cap_pul'],
                   'tex'              : ['$p_{\\\mathrm{O}_2,\\\mathrm{at}}^{r}$', '$p_{\\\mathrm{O}_2,\\\mathrm{v}}^{r}$', '$p_{\\\mathrm{O}_2,\\\mathrm{ar}}^{\\\mathrm{pul}}$', '$p_{\\\mathrm{O}_2,\\\mathrm{ven}}^{\\\mathrm{pul}}$', '$p_{\\\mathrm{O}_2,\\\mathrm{cap}}^{\\\mathrm{pul}}$'],
                   'lines'            : [16, 17, 18, 19, 20]})
    # index 17
    groups.append({'ppCO2_time_pul_r' : ['ppCO2_at_r', 'ppCO2_v_r', 'ppCO2_ar_pul', 'ppCO2_ven_pul', 'ppCO2_cap_pul'],
                   'tex'              : ['$p_{\\\mathrm{CO}_2,\\\mathrm{at}}^{r}$', '$p_{\\\mathrm{CO}_2,\\\mathrm{v}}^{r}$', '$p_{\\\mathrm{CO}_2,\\\mathrm{ar}}^{\\\mathrm{pul}}$', '$p_{\\\mathrm{CO}_2,\\\mathrm{ven}}^{\\\mathrm{pul}}$', '$p_{\\\mathrm{CO}_2,\\\mathrm{cap}}^{\\\mathrm{pul}}$'],
                   'lines'            : [16, 17, 18, 19, 20]})

    # now append all the values again but with suffix PERIODIC, since we want to plot both:
    # values over all heart cycles as well as only for the periodic cycle

    # index 18
    groups.append({'ppO2_time_sys_l_PERIODIC'  : list(groups[14].values())[0],
                   'tex'                       : list(groups[14].values())[1],
                   'lines'                     : list(groups[14].values())[2]})
    # index 19
    groups.append({'ppCO2_time_sys_l_PERIODIC' : list(groups[15].values())[0],
                   'tex'                       : list(groups[15].values())[1],
                   'lines'                     : list(groups[15].values())[2]})
    # index 20
    groups.append({'ppO2_time_pul_r_PERIODIC'  : list(groups[16].values())[0],
                   'tex'                       : list(groups[16].values())[1],
                   'lines'                     : list(groups[16].values())[2]})
    # index 21
    groups.append({'ppCO2_time_pul_r_PERIODIC' : list(groups[17].values())[0],
                   'tex'                       : list(groups[17].values())[1],
                   'lines'                     : list(groups[17].values())[2]})
