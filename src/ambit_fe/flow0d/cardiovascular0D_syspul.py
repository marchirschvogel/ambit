#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import sympy as sp

from .cardiovascular0D import cardiovascular0Dbase
from ..mpiroutines import allgather_vec
from .. import utilities

"""
Systemic and pulmonary closed-loop circulation model, each heart chamber can be treated individually,
either as 0D elastance model, volume or flux coming from a 3D solid, or interface fluxes from a 3D fluid model

18 governing equations:

left heart and systemic circulation:
\begin{align}
&-Q_{\mathrm{at}}^{\ell} = q_{\mathrm{ven}}^{\mathrm{pul}} - q_{\mathrm{v,in}}^{\ell}\\
&\tilde{R}_{\mathrm{v,in}}^{\ell}\,q_{\mathrm{v,in}}^{\ell} = p_{\mathrm{at}}^{\ell}-p_{\mathrm{v}}^{\ell}\\
&-Q_{\mathrm{v}}^{\ell} = q_{\mathrm{v,in}}^{\ell} - q_{\mathrm{v,out}}^{\ell}\\
&\tilde{R}_{\mathrm{v,out}}^{\ell}\,q_{\mathrm{v,out}}^{\ell} = p_{\mathrm{v}}^{\ell}-p_{\mathrm{ar}}^{\mathrm{sys}}\\
&0 = q_{\mathrm{v,out}}^{\ell} - q_{\mathrm{ar,p}}^{\mathrm{sys}}\\
&I_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}q_{\mathrm{ar,p}}^{\mathrm{sys}}}{\mathrm{d}t} + Z_{\mathrm{ar}}^{\mathrm{sys}}\,q_{\mathrm{ar,p}}^{\mathrm{sys}}=p_{\mathrm{ar}}^{\mathrm{sys}}-p_{\mathrm{ar,d}}^{\mathrm{sys}}\\
&C_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ar,d}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar,p}}^{\mathrm{sys}} - q_{\mathrm{ar}}^{\mathrm{sys}}\\
&L_{\mathrm{ar}}^{\mathrm{sys}} \frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{sys}}\,q_{\mathrm{ar}}^{\mathrm{sys}}=p_{\mathrm{ar,d}}^{\mathrm{sys}}-p_{\mathrm{ven}}^{\mathrm{sys}}\\
&C_{\mathrm{ven}}^{\mathrm{sys}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{sys}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{sys}}-q_{\mathrm{ven}}^{\mathrm{sys}}\\
&L_{\mathrm{ven}}^{\mathrm{sys}} \frac{\mathrm{d}q_{\mathrm{ven}}^{\mathrm{sys}}}{\mathrm{d}t} + R_{\mathrm{ven}}^{\mathrm{sys}}\, q_{\mathrm{ven}}^{\mathrm{sys}} = p_{\mathrm{ven}}^{\mathrm{sys}} - p_{\mathrm{at}}^{r}
\end{align}

right heart and pulmonary circulation:
\begin{align}
&-Q_{\mathrm{at}}^{r} = q_{\mathrm{ven}}^{\mathrm{sys}} - q_{\mathrm{v,in}}^{r}\\
&\tilde{R}_{\mathrm{v,in}}^{r}\,q_{\mathrm{v,in}}^{r} = p_{\mathrm{at}}^{r}-p_{\mathrm{v}}^{r}\\
&-Q_{\mathrm{v}}^{r} = q_{\mathrm{v,in}}^{r} - q_{\mathrm{v,out}}^{r}\\
&\tilde{R}_{\mathrm{v,out}}^{r}\,q_{\mathrm{v,out}}^{r} = p_{\mathrm{v}}^{r}-p_{\mathrm{ar}}^{\mathrm{pul}}\\
&C_{\mathrm{ar}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{v,out}}^{r} - q_{\mathrm{ar}}^{\mathrm{pul}}\\
&L_{\mathrm{ar}}^{\mathrm{pul}} \frac{\mathrm{d}q_{\mathrm{ar}}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ar}}^{\mathrm{pul}}\,q_{\mathrm{ar}}^{\mathrm{pul}}=p_{\mathrm{ar}}^{\mathrm{pul}} -p_{\mathrm{ven}}^{\mathrm{pul}}\\
&C_{\mathrm{ven}}^{\mathrm{pul}} \frac{\mathrm{d}p_{\mathrm{ven}}^{\mathrm{pul}}}{\mathrm{d}t} = q_{\mathrm{ar}}^{\mathrm{pul}} - q_{\mathrm{ven}}^{\mathrm{pul}}\\
&L_{\mathrm{ven}}^{\mathrm{pul}} \frac{\mathrm{d}q_{\mathrm{ven}}^{\mathrm{pul}}}{\mathrm{d}t} + R_{\mathrm{ven}}^{\mathrm{pul}}\, q_{\mathrm{ven}}^{\mathrm{pul}}=p_{\mathrm{ven}}^{\mathrm{pul}}-p_{\mathrm{at}}^{\ell}
\end{align}
"""


class cardiovascular0Dsyspul(cardiovascular0Dbase):
    def __init__(
        self,
        params,
        chmodels,
        cq,
        vq,
        valvelaws={
            "av": ["pwlin_pres", 0],
            "mv": ["pwlin_pres", 0],
            "pv": ["pwlin_pres", 0],
            "tv": ["pwlin_pres", 0],
        },
        cormodel=None,
        vadmodel=None,
        init=True,
        ode_par=False,
        comm=None,
    ):
        # initialize base class
        super().__init__(init=init, ode_par=ode_par, comm=comm)

        # parameters
        # circulatory system parameters: resistances (R), compliances (C), inertances (L, I), impedances (Z)
        self.R_ar_sys = params["R_ar_sys"]
        self.C_ar_sys = params["C_ar_sys"]
        self.L_ar_sys = params["L_ar_sys"]
        self.Z_ar_sys = params["Z_ar_sys"]
        self.I_ar_sys = params.get("I_ar_sys", 0.0)
        self.R_ven_sys = params["R_ven_sys"]
        self.C_ven_sys = params["C_ven_sys"]
        self.L_ven_sys = params["L_ven_sys"]
        self.R_ar_pul = params["R_ar_pul"]
        self.C_ar_pul = params["C_ar_pul"]
        self.L_ar_pul = params["L_ar_pul"]
        self.R_ven_pul = params["R_ven_pul"]
        self.C_ven_pul = params["C_ven_pul"]
        self.L_ven_pul = params["L_ven_pul"]

        # ventricular elastances (for 0D ventricles)
        self.E_v_max_l = params["E_v_max_l"]
        self.E_v_min_l = params["E_v_min_l"]
        self.E_v_max_r = params["E_v_max_r"]
        self.E_v_min_r = params["E_v_min_r"]

        # atrial elastances (for 0D atria)
        self.E_at_max_l = params["E_at_max_l"]
        self.E_at_min_l = params["E_at_min_l"]
        self.E_at_max_r = params["E_at_max_r"]
        self.E_at_min_r = params["E_at_min_r"]

        # valve resistances
        self.R_vin_l_min = params["R_vin_l_min"]
        self.R_vin_l_max = params["R_vin_l_max"]
        self.R_vin_r_min = params["R_vin_r_min"]
        self.R_vin_r_max = params["R_vin_r_max"]
        self.R_vout_l_min = params["R_vout_l_min"]
        self.R_vout_l_max = params["R_vout_l_max"]
        self.R_vout_r_min = params["R_vout_r_min"]
        self.R_vout_r_max = params["R_vout_r_max"]

        # valve inertances
        self.L_vin_l = params.get("L_vin_l", 0.0)
        self.L_vin_r = params.get("L_vin_r", 0.0)
        self.L_vout_l = params.get("L_vout_l", 0.0)
        self.L_vout_r = params.get("L_vout_r", 0.0)

        # end-diastolic and end-systolic timings
        self.t_ed = params["t_ed"]
        self.t_es = params["t_es"]
        self.T_cycl = params["T_cycl"]

        # unstressed compartment volumes (for post-processing)
        self.V_at_l_u = params.get("V_at_l_u", 0.0)
        self.V_at_r_u = params.get("V_at_r_u", 0.0)
        self.V_v_l_u = params.get("V_v_l_u", 0.0)
        self.V_v_r_u = params.get("V_v_r_u", 0.0)
        self.V_ar_sys_u = params.get("V_ar_sys_u", 0.0)
        self.V_ar_pul_u = params.get("V_ar_pul_u", 0.0)
        self.V_ven_sys_u = params.get("V_ven_sys_u", 0.0)
        self.V_ven_pul_u = params.get("V_ven_pul_u", 0.0)

        self.params = params

        self.chmodels = chmodels
        self.valvelaws = valvelaws

        self.cormodel = cormodel
        self.vadmodel = vadmodel

        # number of systemic venous inflows (to right atrium)
        self.vs = self.chmodels["ra"].get("num_inflows", 1)
        # number of pulmonary venous inflows (to left atrium)
        self.vp = self.chmodels["la"].get("num_inflows", 1)

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
        self.numdof = 16 + self.vs + self.vp

        self.elastarrays = [[]] * 4

        self.si, self.switch_V = [0] * 5, [1] * 5  # default values

        self.vindex_ch = [
            3,
            12 + self.vs,
            1,
            10 + self.vs,
            4,
        ]  # coupling variable indices (decreased by 1 for pressure coupling!)
        self.vname, self.cname = (
            ["p_v_l", "p_v_r", "p_at_l", "p_at_r", "p_ar_sys"],
            [],
        )

        # set those ids which are relevant for monolithic direct coupling
        self.v_ids, self.c_ids = [], []
        self.cindex_ch = [2, 11 + self.vs, 0, 9 + self.vs]

        self.switch_cor, self.switch_vad = 0, 0

        if self.cormodel is not None:
            # additional venous inflow is added by coronary model, so make sure that vs is one less if user requests a 3D RA model
            if "num_inflows" in self.chmodels["ra"].keys() and self.vs > 1:
                self.vs -= 1

            # initialize coronary circulation model
            if self.cormodel == "ZCRp_CRd":
                from .cardiovascular0D_coronary import coronary_circ_ZCRp_CRd

                self.corcirc = coronary_circ_ZCRp_CRd(self.params, self.varmap, self.auxmap, self.vs)
            elif self.cormodel == "ZCRp_CRd_lr":
                from .cardiovascular0D_coronary import (
                    coronary_circ_ZCRp_CRd_lr,
                )

                self.corcirc = coronary_circ_ZCRp_CRd_lr(self.params, self.varmap, self.auxmap, self.vs)
            else:
                raise NameError("Unknown coronary circulation model!")

            self.switch_cor = 1

            self.doff_cor = self.numdof
            self.numdof += self.corcirc.ndcor

        if self.vadmodel is not None:
            # initialize VAD model
            if self.vadmodel == "lvad":
                from .cardiovascular0D_vad import vad_circ

                self.vadcirc = vad_circ(self.params, self.varmap, self.auxmap)
            else:
                raise NameError("Unknown VAD model!")

            self.switch_vad = 1

            self.doff_vad = self.numdof
            self.numdof += self.vadcirc.ndvad

        self.set_solve_arrays()

    def evaluate(self, x, t, df=None, f=None, dK=None, K=None, c=[], y=[], a=None):
        fnc = self.evaluate_chamber_state(y, t)

        cardiovascular0Dbase.evaluate(self, x, t, df, f, dK, K, c, y, a, fnc)

    def equation_map(self):
        # variable map
        self.varmap["q_vin_l"] = 0 + self.si[2]
        self.varmap[self.vname[2]] = 1 - self.si[2]
        self.varmap["q_vout_l"] = 2 + self.si[0]
        self.varmap[self.vname[0]] = 3 - self.si[0]
        self.varmap[self.vname[4]] = 4
        self.varmap["q_arp_sys"] = 5
        self.varmap["p_ard_sys"] = 6
        self.varmap["q_ar_sys"] = 7
        self.varmap["p_ven_sys"] = 8
        for n in range(self.vs):
            self.varmap["q_ven" + str(n + 1) + "_sys"] = 9 + n
        self.varmap["q_vin_r"] = 9 + self.vs + self.si[3]
        self.varmap[self.vname[3]] = 10 + self.vs - self.si[3]
        self.varmap["q_vout_r"] = 11 + self.vs + self.si[1]
        self.varmap[self.vname[1]] = 12 + self.vs - self.si[1]
        self.varmap["p_ar_pul"] = 13 + self.vs
        self.varmap["q_ar_pul"] = 14 + self.vs
        self.varmap["p_ven_pul"] = 15 + self.vs
        for n in range(self.vp):
            self.varmap["q_ven" + str(n + 1) + "_pul"] = 16 + self.vs + n

        q_ven_sys_, q_ven_pul_ = [], []
        p_at_l_i_, p_at_r_i_ = [], []

        self.t_ = sp.Symbol("t_")
        q_vin_l_ = sp.Symbol("q_vin_l_")
        for n in range(self.vp):
            p_at_l_i_.append(sp.Symbol("p_at_l_i" + str(n + 1) + "_"))
        p_at_l_o1_ = sp.Symbol("p_at_l_o1_")
        q_vout_l_ = sp.Symbol("q_vout_l_")
        p_v_l_i1_, p_v_l_o1_ = sp.Symbol("p_v_l_i1_"), sp.Symbol("p_v_l_o1_")
        p_v_l_o2_ = sp.Symbol("p_v_l_o2_")  # for a VAD model
        p_ar_sys_i1_, p_ar_sys_o1_, p_ar_sys_o2_, p_ar_sys_o3_ = (
            sp.Symbol("p_ar_sys_i1_"),
            sp.Symbol("p_ar_sys_o1_"),
            sp.Symbol("p_ar_sys_o2_"),
            sp.Symbol("p_ar_sys_o3_"),
        )
        p_ar_sys_o4_ = sp.Symbol("p_ar_sys_o4_")  # for a VAD model
        q_arp_sys_ = sp.Symbol("q_arp_sys_")
        p_ard_sys_ = sp.Symbol("p_ard_sys_")
        q_ar_sys_ = sp.Symbol("q_ar_sys_")
        p_ven_sys_ = sp.Symbol("p_ven_sys_")
        for n in range(self.vs):
            q_ven_sys_.append(sp.Symbol("q_ven" + str(n + 1) + "_sys_"))
        q_vin_r_ = sp.Symbol("q_vin_r_")
        for n in range(self.vs + self.switch_cor):
            p_at_r_i_.append(sp.Symbol("p_at_r_i" + str(n + 1) + "_"))
        p_at_r_o1_ = sp.Symbol("p_at_r_o1_")
        q_vout_r_ = sp.Symbol("q_vout_r_")
        p_v_r_i1_, p_v_r_o1_ = sp.Symbol("p_v_r_i1_"), sp.Symbol("p_v_r_o1_")
        p_ar_pul_ = sp.Symbol("p_ar_pul_")
        q_ar_pul_ = sp.Symbol("q_ar_pul_")
        p_ven_pul_ = sp.Symbol("p_ven_pul_")
        for n in range(self.vp):
            q_ven_pul_.append(sp.Symbol("q_ven" + str(n + 1) + "_pul_"))
        VQ_v_l_ = sp.Symbol("VQ_v_l_")
        VQ_v_r_ = sp.Symbol("VQ_v_r_")
        VQ_at_l_ = sp.Symbol("VQ_at_l_")
        VQ_at_r_ = sp.Symbol("VQ_at_r_")
        # aortic root/ascending aortic volume (for 3D flow analysis)
        VQ_aort_sys_ = sp.Symbol("VQ_aort_sys_")

        E_v_l_ = sp.Symbol("E_v_l_")
        E_v_r_ = sp.Symbol("E_v_r_")
        E_at_l_ = sp.Symbol("E_at_l_")
        E_at_r_ = sp.Symbol("E_at_r_")

        # dofs to differentiate w.r.t.
        self.x_[self.varmap["q_vin_l"]] = q_vin_l_
        self.x_[self.varmap[self.vname[2]]] = p_at_l_i_[0]
        self.x_[self.varmap["q_vout_l"]] = q_vout_l_
        self.x_[self.varmap[self.vname[0]]] = p_v_l_i1_
        self.x_[self.varmap[self.vname[4]]] = p_ar_sys_i1_
        self.x_[self.varmap["q_arp_sys"]] = q_arp_sys_
        self.x_[self.varmap["p_ard_sys"]] = p_ard_sys_
        self.x_[self.varmap["q_ar_sys"]] = q_ar_sys_
        self.x_[self.varmap["p_ven_sys"]] = p_ven_sys_
        for n in range(self.vs):
            self.x_[self.varmap["q_ven" + str(n + 1) + "_sys"]] = q_ven_sys_[n]
        self.x_[self.varmap["q_vin_r"]] = q_vin_r_
        self.x_[self.varmap[self.vname[3]]] = p_at_r_i_[0]
        self.x_[self.varmap["q_vout_r"]] = q_vout_r_
        self.x_[self.varmap[self.vname[1]]] = p_v_r_i1_
        self.x_[self.varmap["p_ar_pul"]] = p_ar_pul_
        self.x_[self.varmap["q_ar_pul"]] = q_ar_pul_
        self.x_[self.varmap["p_ven_pul"]] = p_ven_pul_
        for n in range(self.vp):
            self.x_[self.varmap["q_ven" + str(n + 1) + "_pul"]] = q_ven_pul_[n]

        # set chamber dicts
        chdict_lv = {
            "VQ": VQ_v_l_,
            "pi1": p_v_l_i1_,
            "po1": p_v_l_o1_,
            "po2": p_v_l_o2_,
        }
        chdict_rv = {"VQ": VQ_v_r_, "pi1": p_v_r_i1_, "po1": p_v_r_o1_}
        chdict_la = {"VQ": VQ_at_l_, "po1": p_at_l_o1_}
        for n in range(self.vp):
            chdict_la["pi" + str(n + 1)] = p_at_l_i_[n]
        chdict_ra = {"VQ": VQ_at_r_, "po1": p_at_r_o1_}
        for n in range(self.vs + self.switch_cor):
            chdict_ra["pi" + str(n + 1)] = p_at_r_i_[n]
        # aortic root/ascending aortic compartment dict, 1 inflow and 3 outflows (one into arch, two into coronaries)
        chdict_ao = {
            "VQ": VQ_aort_sys_,
            "pi1": p_ar_sys_i1_,
            "po1": p_ar_sys_o1_,
            "po2": p_ar_sys_o2_,
            "po3": p_ar_sys_o3_,
            "po4": p_ar_sys_o4_,
        }

        # set coupling states and variables (e.g., express V in terms of p and E in case of elastance models, ...)
        self.set_coupling_state("lv", chdict_lv, [E_v_l_])
        self.set_coupling_state("rv", chdict_rv, [E_v_r_])
        self.set_coupling_state("la", chdict_la, [E_at_l_])
        self.set_coupling_state("ra", chdict_ra, [E_at_r_])
        # aortic root/ascending aortic compartment
        self.set_coupling_state("ao", chdict_ao, [])

        # feed back modified dicts to chamber variables
        VQ_v_l_, p_v_l_i1_, p_v_l_o1_, p_v_l_o2_ = (
            chdict_lv["VQ"],
            chdict_lv["pi1"],
            chdict_lv["po1"],
            chdict_lv["po2"],
        )
        VQ_v_r_, p_v_r_i1_, p_v_r_o1_ = (
            chdict_rv["VQ"],
            chdict_rv["pi1"],
            chdict_rv["po1"],
        )
        VQ_at_l_, p_ati1_l_, p_at_l_o1_ = (
            chdict_la["VQ"],
            chdict_la["pi1"],
            chdict_la["po1"],
        )
        for n in range(self.vp):
            p_at_l_i_[n] = chdict_la["pi" + str(n + 1)]
        VQ_at_r_, p_ati1_r_, p_at_r_o1_ = (
            chdict_ra["VQ"],
            chdict_ra["pi1"],
            chdict_ra["po1"],
        )
        for n in range(self.vs + self.switch_cor):
            p_at_r_i_[n] = chdict_ra["pi" + str(n + 1)]
        # aortic root/ascending aortic compartment
        (
            VQ_aort_sys_,
            p_ar_sys_i1_,
            p_ar_sys_o1_,
            p_ar_sys_o2_,
            p_ar_sys_o3_,
            p_ar_sys_o4_,
        ) = (
            chdict_ao["VQ"],
            chdict_ao["pi1"],
            chdict_ao["po1"],
            chdict_ao["po2"],
            chdict_ao["po3"],
            chdict_ao["po4"],
        )

        # add coronary circulation equations
        if self.cormodel is not None:
            q_arcor_sys_in_, q_vencor_sys_out_ = self.corcirc.equation_map(
                self.doff_cor,
                len(self.c_) + 8,
                self.x_,
                self.a_,
                self.df_,
                self.f_,
                [p_ar_sys_o1_, p_ar_sys_o2_],
                p_v_l_o1_,
                p_at_r_i_[-1],
            )
        else:
            q_arcor_sys_in_, q_vencor_sys_out_ = [0], 0

        # add VAD circulation equations
        if self.vadmodel is not None:
            q_vad_in_, q_vad_out_ = self.vadcirc.equation_map(
                self.doff_vad,
                len(self.c_) + 8,
                self.x_,
                self.a_,
                self.df_,
                self.f_,
                p_v_l_o2_,
                p_ar_sys_o4_,
            )
        else:
            q_vad_in_, q_vad_out_ = 0, 0

        # set valve laws - resistive part of q(p) relationship of momentum equation
        vl_mv_, R_vin_l_ = self.valvelaw(
            p_at_l_o1_,
            p_v_l_i1_,
            self.R_vin_l_min,
            self.R_vin_l_max,
            self.valvelaws["mv"],
            self.t_es,
            self.t_ed,
        )
        vl_av_, R_vout_l_ = self.valvelaw(
            p_v_l_o1_,
            p_ar_sys_i1_,
            self.R_vout_l_min,
            self.R_vout_l_max,
            self.valvelaws["av"],
            self.t_ed,
            self.t_es,
        )
        vl_tv_, R_vin_r_ = self.valvelaw(
            p_at_r_o1_,
            p_v_r_i1_,
            self.R_vin_r_min,
            self.R_vin_r_max,
            self.valvelaws["tv"],
            self.t_es,
            self.t_ed,
        )
        vl_pv_, R_vout_r_ = self.valvelaw(
            p_v_r_o1_,
            p_ar_pul_,
            self.R_vout_r_min,
            self.R_vout_r_max,
            self.valvelaws["pv"],
            self.t_ed,
            self.t_es,
        )

        # parallel venous resistances and inertances:
        # assume that the total venous resistance/inertance distributes equally over all systemic / pulmonary veins that enter the right / left atrium
        # resistance/inertance in parallel: 1/R_total = 1/R_1 + 1/R_2 + ... + 1/R_n, 1/L_total = 1/L_1 + 1/L_2 + ... + 1/L_n
        # let's say: R_1 = R_2 = ... = R_n, L_1 = L_2 = ... = L_n

        R_ven_sys, L_ven_sys, R_ven_pul, L_ven_pul = [], [], [], []
        for n in range(self.vs):
            R_ven_sys.append(self.vs * self.R_ven_sys)
            L_ven_sys.append(self.vs * self.L_ven_sys)

        for n in range(self.vp):
            R_ven_pul.append(self.vp * self.R_ven_pul)
            L_ven_pul.append(self.vp * self.L_ven_pul)

        # df part of rhs contribution (df - df_old)/dt
        self.df_[0] = VQ_at_l_ * self.switch_V[2]  # left atrium volume rate
        self.df_[1] = (self.L_vin_l / R_vin_l_) * q_vin_l_  # mitral valve inertia
        self.df_[2] = VQ_v_l_ * self.switch_V[0]  # left ventricle volume rate
        self.df_[3] = (self.L_vout_l / R_vout_l_) * q_vout_l_  # aortic valve inertia
        self.df_[4] = 0.0
        self.df_[5] = (self.I_ar_sys / self.Z_ar_sys) * q_arp_sys_  # aortic root inertia
        self.df_[6] = self.C_ar_sys * p_ard_sys_  # systemic arterial volume rate
        self.df_[7] = (self.L_ar_sys / self.R_ar_sys) * q_ar_sys_  # systemic arterial inertia
        self.df_[8] = self.C_ven_sys * p_ven_sys_  # systemic venous volume rate
        for n in range(self.vs):
            self.df_[9 + n] = (L_ven_sys[n] / R_ven_sys[n]) * q_ven_sys_[n]  # systemic venous inertia
            # -----------------------------------------------------------
        self.df_[9 + self.vs] = VQ_at_r_ * self.switch_V[3]  # right atrium volume rate
        self.df_[10 + self.vs] = (self.L_vin_r / R_vin_r_) * q_vin_r_  # tricuspid valve inertia
        self.df_[11 + self.vs] = VQ_v_r_ * self.switch_V[1]  # right ventricle volume rate
        self.df_[12 + self.vs] = (self.L_vout_r / R_vout_r_) * q_vout_r_  # pulmonary valve inertia
        self.df_[13 + self.vs] = self.C_ar_pul * p_ar_pul_  # pulmonary arterial volume rate
        self.df_[14 + self.vs] = (self.L_ar_pul / self.R_ar_pul) * q_ar_pul_  # pulmonary arterial inertia
        self.df_[15 + self.vs] = self.C_ven_pul * p_ven_pul_  # pulmonary venous volume rate
        for n in range(self.vp):
            self.df_[16 + self.vs + n] = (L_ven_pul[n] / R_ven_pul[n]) * q_ven_pul_[n]  # pulmonary venous inertia

        # f part of rhs contribution theta * f + (1-theta) * f_old
        self.f_[0] = -sum(q_ven_pul_) + q_vin_l_ - (1 - self.switch_V[2]) * VQ_at_l_  # left atrium flow balance
        self.f_[1] = vl_mv_ + q_vin_l_  # mitral valve momentum
        self.f_[2] = (
            -q_vin_l_ + q_vout_l_ - (1 - self.switch_V[0]) * VQ_v_l_ + self.switch_vad * q_vad_in_
        )  # left ventricle flow balance
        self.f_[3] = vl_av_ + q_vout_l_  # aortic valve momentum
        self.f_[4] = (
            -q_vout_l_
            + q_arp_sys_
            + self.switch_cor * sum(q_arcor_sys_in_)
            - self.switch_vad * q_vad_out_
            - VQ_aort_sys_
        )  # aortic root flow balance
        self.f_[5] = (p_ard_sys_ - p_ar_sys_o3_) / self.Z_ar_sys + q_arp_sys_  # aortic root momentum
        self.f_[6] = -q_arp_sys_ + q_ar_sys_  # systemic arterial flow balance
        self.f_[7] = (p_ven_sys_ - p_ard_sys_) / self.R_ar_sys + q_ar_sys_  # systemic arterial momentum
        self.f_[8] = -q_ar_sys_ + sum(q_ven_sys_)  # systemic venous flow balance
        for n in range(self.vs):
            self.f_[9 + n] = (p_at_r_i_[n] - p_ven_sys_) / R_ven_sys[n] + q_ven_sys_[n]  # systemic venous momentum
            # -----------------------------------------------------------
        self.f_[9 + self.vs] = (
            -sum(q_ven_sys_) - self.switch_cor * q_vencor_sys_out_ + q_vin_r_ - (1 - self.switch_V[3]) * VQ_at_r_
        )  # right atrium flow balance
        self.f_[10 + self.vs] = vl_tv_ + q_vin_r_  # tricuspid valve momentum
        self.f_[11 + self.vs] = -q_vin_r_ + q_vout_r_ - (1 - self.switch_V[1]) * VQ_v_r_  # right ventricle flow balance
        self.f_[12 + self.vs] = vl_pv_ + q_vout_r_  # pulmonary valve momentum
        self.f_[13 + self.vs] = -q_vout_r_ + q_ar_pul_  # pulmonary arterial flow balance
        self.f_[14 + self.vs] = (p_ven_pul_ - p_ar_pul_) / self.R_ar_pul + q_ar_pul_  # pulmonary arterial momentum
        self.f_[15 + self.vs] = -q_ar_pul_ + sum(q_ven_pul_)  # pulmonary venous flow balance
        for n in range(self.vp):
            self.f_[16 + self.vs + n] = (p_at_l_i_[n] - p_ven_pul_) / R_ven_pul[n] + q_ven_pul_[
                n
            ]  # pulmonary venous momentum

        # setup auxiliary variable map
        # coupling variables, 0D chamber volumes, compartment volumes, other shady quantities...
        nc = len(self.c_)
        for i in range(nc):
            self.auxmap[self.cname[i]] = i
        if (
            self.chmodels["lv"]["type"] == "0D_elast"
            or self.chmodels["lv"]["type"] == "prescribed"
            or self.chmodels["lv"]["type"] == "0D_elast_prescr"
        ):
            self.auxmap["V_v_l"] = nc + 0
        if (
            self.chmodels["rv"]["type"] == "0D_elast"
            or self.chmodels["rv"]["type"] == "prescribed"
            or self.chmodels["rv"]["type"] == "0D_elast_prescr"
        ):
            self.auxmap["V_v_r"] = nc + 1
        if (
            self.chmodels["la"]["type"] == "0D_elast"
            or self.chmodels["la"]["type"] == "prescribed"
            or self.chmodels["la"]["type"] == "0D_elast_prescr"
        ):
            self.auxmap["V_at_l"] = nc + 2
        if (
            self.chmodels["ra"]["type"] == "0D_elast"
            or self.chmodels["ra"]["type"] == "prescribed"
            or self.chmodels["ra"]["type"] == "0D_elast_prescr"
        ):
            self.auxmap["V_at_r"] = nc + 3
        self.auxmap["V_ar_sys"] = nc + 4
        self.auxmap["V_ven_sys"] = nc + 5
        self.auxmap["V_ar_pul"] = nc + 6
        self.auxmap["V_ven_pul"] = nc + 7

        # populate auxiliary variable vector
        for i in range(nc):
            self.a_[i] = self.c_[i]
        self.a_[nc + 0] = VQ_v_l_ * self.switch_V[0]
        self.a_[nc + 1] = VQ_v_r_ * self.switch_V[1]
        self.a_[nc + 2] = VQ_at_l_ * self.switch_V[2]
        self.a_[nc + 3] = VQ_at_r_ * self.switch_V[3]
        self.a_[nc + 4] = self.C_ar_sys * p_ard_sys_ + self.V_ar_sys_u
        self.a_[nc + 5] = self.C_ven_sys * p_ven_sys_ + self.V_ven_sys_u
        self.a_[nc + 6] = self.C_ar_pul * p_ar_pul_ + self.V_ar_pul_u
        self.a_[nc + 7] = self.C_ven_pul * p_ven_pul_ + self.V_ven_pul_u

    def initialize(self, var, iniparam):
        var[self.varmap["q_vin_l"]] = iniparam["q_vin_l_0"]
        var[self.varmap[self.vname[2]]] = iniparam[self.vname[2] + "_0"]
        var[self.varmap["q_vout_l"]] = iniparam["q_vout_l_0"]
        var[self.varmap[self.vname[0]]] = iniparam[self.vname[0] + "_0"]
        var[self.varmap[self.vname[4]]] = iniparam[self.vname[4] + "_0"]
        try:
            var[self.varmap["q_arp_sys"]] = iniparam["q_arp_sys_0"]
        except:
            var[self.varmap["q_arp_sys"]] = iniparam["q_ar_sys_0"]
        try:
            var[self.varmap["p_ard_sys"]] = iniparam["p_ard_sys_0"]
        except:
            var[self.varmap["p_ard_sys"]] = iniparam["p_ar_sys_0"]
        var[self.varmap["q_ar_sys"]] = iniparam["q_ar_sys_0"]
        var[self.varmap["p_ven_sys"]] = iniparam["p_ven_sys_0"]
        for n in range(self.vs):
            try:
                var[self.varmap["q_ven" + str(n + 1) + "_sys"]] = iniparam["q_ven" + str(n + 1) + "_sys_0"]
            except:
                var[self.varmap["q_ven" + str(n + 1) + "_sys"]] = iniparam["q_ven_sys_0"]
        var[self.varmap["q_vin_r"]] = iniparam["q_vin_r_0"]
        var[self.varmap[self.vname[3]]] = iniparam[self.vname[3] + "_0"]
        var[self.varmap["q_vout_r"]] = iniparam["q_vout_r_0"]
        var[self.varmap[self.vname[1]]] = iniparam[self.vname[1] + "_0"]
        var[self.varmap["p_ar_pul"]] = iniparam["p_ar_pul_0"]
        var[self.varmap["q_ar_pul"]] = iniparam["q_ar_pul_0"]
        var[self.varmap["p_ven_pul"]] = iniparam["p_ven_pul_0"]
        for n in range(self.vp):
            try:
                var[self.varmap["q_ven" + str(n + 1) + "_pul"]] = iniparam["q_ven" + str(n + 1) + "_pul_0"]
            except:
                var[self.varmap["q_ven" + str(n + 1) + "_pul"]] = iniparam["q_ven_pul_0"]

        if self.cormodel is not None:
            self.corcirc.initialize(var, iniparam)

        if self.vadmodel is not None:
            self.vadcirc.initialize(var, iniparam)

    def check_periodic(self, varTc, varTc_old, auxTc, auxTc_old, eps, check, cyclerr):
        vals = []

        if check[0] == "allvar":
            var_ids, aux_ids = list(self.varmap.values()), []

        elif check[0] == "allvaraux":
            var_ids, aux_ids = (
                list(self.varmap.values()),
                list(self.auxmap.values()),
            )

        elif check[0] == "pQvar":
            var_ids, aux_ids = (
                [
                    self.varmap[self.vname[2]],
                    self.varmap[self.vname[0]],
                    self.varmap[self.vname[4]],
                    self.varmap["p_ard_sys"],
                    self.varmap["p_ven_sys"],
                    self.varmap[self.vname[3]],
                    self.varmap[self.vname[1]],
                    self.varmap["p_ar_pul"],
                    self.varmap["p_ven_pul"],
                ],
                [],
            )

        elif check[0] == "specific":
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
        for i in range(len(varTc.array)):
            if i in var_ids:
                vals.append(math.fabs((varTc.array[i] - varTc_old.array[i]) / max(1.0, math.fabs(varTc_old.array[i]))))

        for i in range(len(auxTc)):
            if i in aux_ids:
                vals.append(math.fabs((auxTc[i] - auxTc_old[i]) / max(1.0, math.fabs(auxTc_old[i]))))

        cyclerr[0] = max(vals)

        if cyclerr[0] <= eps:
            is_periodic = True
        else:
            is_periodic = False

        return is_periodic

    def print_to_screen(self, var, aux):
        if self.ode_parallel:
            var_arr = allgather_vec(var, self.comm)
        else:
            var_arr = var.array

        nc = len(self.c_)

        utilities.print_status("Output of 0D vascular model (syspul):", self.comm)

        for i in range(nc):
            utilities.print_status(
                "{:<9s}{:<3s}{:<10.3f}".format(self.cname[i], " = ", aux[self.auxmap[self.cname[i]]]),
                self.comm,
            )

        utilities.print_status(
            "{:<9s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}".format(
                self.vname[2],
                " = ",
                var_arr[self.varmap[self.vname[2]]],
                "   ",
                self.vname[3],
                " = ",
                var_arr[self.varmap[self.vname[3]]],
            ),
            self.comm,
        )
        utilities.print_status(
            "{:<9s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}".format(
                self.vname[0],
                " = ",
                var_arr[self.varmap[self.vname[0]]],
                "   ",
                self.vname[1],
                " = ",
                var_arr[self.varmap[self.vname[1]]],
            ),
            self.comm,
        )

        utilities.print_status(
            "{:<9s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}".format(
                self.vname[4],
                " = ",
                var_arr[self.varmap[self.vname[4]]],
                "   ",
                "p_ar_pul",
                " = ",
                var_arr[self.varmap["p_ar_pul"]],
            ),
            self.comm,
        )
        utilities.print_status(
            "{:<9s}{:<3s}{:<10.3f}{:<3s}{:<9s}{:<3s}{:<10.3f}".format(
                "p_ven_sys",
                " = ",
                var_arr[self.varmap["p_ven_sys"]],
                "   ",
                "p_ven_pul",
                " = ",
                var_arr[self.varmap["p_ven_pul"]],
            ),
            self.comm,
        )

        if self.cormodel is not None:
            self.corcirc.print_to_screen(var_arr, aux, self.comm)
        if self.vadmodel is not None:
            self.vadcirc.print_to_screen(var_arr, aux, self.comm)
