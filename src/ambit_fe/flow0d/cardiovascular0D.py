#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import sys, time
import numpy as np
import sympy as sp

from ..mpiroutines import allgather_vec_entry
from ..oderoutines import ode


class cardiovascular0Dbase(ode):

    def __init__(self, init=True, comm=None):

        # initialize base class
        super().__init__(init=init, comm=comm)

        self.T_cycl = 0 # duration of one cardiac cycle (gets overridden by derived syspul* classes)
        self.off_io = 0 # offsets for in-/outflows for coupling indices


    # check for cardiac cycle periodicity
    def cycle_check(self, var, varTc, varTc_old, aux, auxTc, auxTc_old, t, cycle, cyclerr, eps_periodic, check=['allvar'], inioutpath=None, nm='', induce_pert_after_cycl=-1):

        if isinstance(varTc, np.ndarray): vs, ve = 0, len(varTc)
        else: vs, ve = var.getOwnershipRange()

        is_periodic = False

        if self.T_cycl > 0. and np.isclose(t, self.T_cycl):

            varTc[vs:ve] = var[vs:ve]
            auxTc[:] = aux[:]

            if check[0] is not None: is_periodic = self.check_periodic(varTc, varTc_old, auxTc, auxTc_old, eps_periodic, check, cyclerr)

            # definitely should not be True if we've not yet surpassed the "disease induction" cycle
            if cycle[0] <= induce_pert_after_cycl:
                is_periodic = False

            # write "periodic" initial conditions in case we want to restart from this model in another simulation
            if is_periodic and inioutpath is not None:
                self.write_initial(inioutpath, nm, varTc_old, varTc, auxTc_old, auxTc)

            varTc_old[vs:ve] = varTc[vs:ve]
            auxTc_old[:] = auxTc[:]

            # update cycle counter
            cycle[0] += 1

        return is_periodic


    # some perturbations/diseases we want to simulate (mr: mitral regurgitation, ms: mitral stenosis, ar: aortic regurgitation, as: aortic stenosis)
    def induce_perturbation(self, perturb_type, perturb_factor):

        if perturb_type=='mr': self.R_vin_l_max *= perturb_factor
        if perturb_type=='ms': self.R_vin_l_min *= perturb_factor
        if perturb_type=='ar': self.R_vout_l_max *= perturb_factor
        if perturb_type=='as': self.R_vout_l_min *= perturb_factor

        # arrays need re-initialization, expressions have to be re-set
        self.setup_arrays(), self.set_compartment_interfaces()
        self.equation_map(), self.set_stiffness(), self.lambdify_expressions()


    # set pressure function for 3D FEM model
    def set_pressure_fem(self, var, ids, pr0D, p0Da):

        # set pressure functions
        for i in range(len(ids)):
            pr0D.val = -allgather_vec_entry(var, ids[i], self.comm)
            p0Da[i].interpolate(pr0D.evaluate)


    # set valve q(p) relationship
    def valvelaw(self, p, popen, Rmin, Rmax, vparams, topen, tclose):

        if vparams[0]=='pwlin_pres': # piecewise linear with resistance depending on pressure difference
            R = sp.Piecewise( (Rmax, p < popen), (Rmin, p >= popen) )
            vl = (popen - p) / R
        elif vparams[0]=='pwlin_time': # piecewise linear with resistance depending on timing
            if topen > tclose: R = sp.Piecewise( (Rmax, sp.And(self.t_ < topen, self.t_ >= tclose)), (Rmin, sp.Or(self.t_ >= topen, self.t_ < tclose)) )
            else:              R = sp.Piecewise( (Rmax, sp.Or(self.t_ < topen, self.t_ >= tclose)), (Rmin, sp.And(self.t_ >= topen, self.t_ < tclose)) )
            vl = (popen - p) / R
        elif vparams[0]=='smooth_pres_resistance': # smooth resistance value
            R = 0.5*(Rmax - Rmin)*(sp.tanh((popen - p)/vparams[-1]) + 1.) + Rmin
            vl = (popen - p) / R
        elif vparams[0]=='smooth_pres_momentum': # smooth q(p) relationship
            # interpolation by cubic spline in epsilon interval
            p0 = (popen-vparams[-1]/2. - popen)/Rmax
            p1 = (popen+vparams[-1]/2. - popen)/Rmin
            m0 = 1./Rmax
            m1 = 1./Rmin
            s = (p - (popen-vparams[-1]/2.))/vparams[-1]
            # spline ansatz functions
            h00 = 2.*s**3. - 3*s**2. + 1.
            h01 = -2.*s**3. + 3*s**2.
            h10 = s**3. - 2.*s**2. + s
            h11 = s**3. - s**2.
            # spline
            c = h00*p0 + h10*m0*vparams[-1] + h01*p1 + h11*m1*vparams[-1]
            vl = sp.Piecewise( ((popen - p)/Rmax, p < popen-vparams[-1]/2), (-c, sp.And(p >= popen-vparams[-1]/2., p < popen+vparams[-1]/2.)), ((popen - p)/Rmin, p >= popen+vparams[-1]/2.) )
        elif vparams[0]=='pw_pres_regurg':
            vl = sp.Piecewise( (vparams[1]*vparams[2]*sp.sqrt(popen - p), p < popen), ((popen - p) / Rmin, p >= popen) )
        else:
            raise NameError("Unknown valve law %s!" % (vparams[0]))

        vlaw = vl
        if popen is not sp.S.Zero:
            res = 1./sp.diff(vl,popen)
        else:
            res = sp.S.One

        return vlaw, res


    # set compartment interfaces according to case and coupling quantity (can be volume, flux, or pressure)
    def set_compartment_interfaces(self):

        # first get the number of in- and out-flows (defaults: 1) in case of 3D-0D fluid coupling
        # can be zero for a 3D chamber that is linked to another 3D one (e.g. LA and LV)
        num_infl, num_outfl = [1]*5, [1]*5
        for i, ch in enumerate(['lv','rv','la','ra', 'ao']):
            try: num_infl[i] = self.chmodels[ch]['num_inflows']
            except: num_infl[i] = 1
            try: num_outfl[i] = self.chmodels[ch]['num_outflows'] # actually not used so far...
            except: num_outfl[i] = 1

        # loop over chambers
        for i, ch in enumerate(['lv','rv','la','ra', 'ao']):

            # name mapping
            if ch == 'lv': chn = 'v_l'
            if ch == 'rv': chn = 'v_r'
            if ch == 'la': chn = 'at_l'
            if ch == 'ra': chn = 'at_r'
            if ch == 'ao': chn = 'aort_sys'

            # now the in- and out-flow indices in case of 3D-0D fluid coupling
            if ch == 'lv': # allow 1 in-flow, 1 ouf-flow for now...
                ind_i = [0] # q_vin_l
                ind_o = [3] # q_vout_l
                if self.chmodels['la']['type']=='3D_fluid': ind_i[0] += 1
            if ch == 'rv': # allow 1 in-flow, 1 ouf-flow for now...
                ind_i = [9+num_infl[3]] # q_vin_r
                ind_o = [11+num_infl[3]] # q_vout_r
                if self.chmodels['ra']['type']=='3D_fluid': ind_i[0] += 1
            if ch == 'la': # allow 5 in-flows, 1 ouf-flow for now...
                ind_i = [16+num_infl[3],17+num_infl[3],18+num_infl[3],19+num_infl[3],20+num_infl[3]] # q_ven,1_pul, ..., q_ven,5_pul
                ind_o = [1] # q_vin_l
            if ch == 'ra': # allow 5 in-flows, 1 ouf-flow for now...
                ind_i = [9+num_infl[3],10+num_infl[3],11+num_infl[3],12+num_infl[3],13+num_infl[3],14+num_infl[3]] # q_ven,1_sys, ..., q_ven,5_sys
                ind_o = [10+num_infl[3]] # q_vin_r
            if ch == 'ao': # allow 1 in-flow, 3 ouf-flows for now...
                ind_i = [2] # q_vout_l
                if self.cormodel: ind_o = [16+num_infl[3]+num_infl[2], 20+num_infl[3]+num_infl[2], 5] # q_corp_sys_l_in, q_corp_sys_r_in, q_arp_sys
                else: ind_o = [5] # q_arp_sys

            if self.chmodels[ch]['type']=='0D_elast' or self.chmodels[ch]['type']=='0D_elast_prescr':
                self.switch_V[i] = 1

            elif self.chmodels[ch]['type']=='0D_rigid':
                self.switch_V[i] = 0

            elif self.chmodels[ch]['type']=='0D_prescr':
                if self.cq[i] == 'volume':
                    assert(self.vq[i]=='pressure')
                    self.switch_V[i] = 1
                    self.cname.append('V_'+chn)
                elif self.cq[i] == 'flux':
                    assert(self.vq[i]=='pressure')
                    self.switch_V[i] = 0
                    self.cname.append('Q_'+chn)
                elif self.cq[i] == 'pressure':
                    if self.vq[i] == 'volume':
                        self.switch_V[i], self.vname[i] = 1, 'V_'+chn
                    elif self.vq[i] == 'flux':
                        self.switch_V[i], self.vname[i] = 0, 'Q_'+chn
                    else:
                        raise ValueError("Variable quantity has to be volume or flux!")
                    self.cname.append('p_'+chn)
                    self.si[i] = 1 # switch indices of pressure / outflux
                else:
                    raise NameError("Unknown coupling quantity!")

            elif self.chmodels[ch]['type']=='3D_solid':
                if self.cq[i] == 'volume':
                    assert(self.vq[i]=='pressure')
                    self.v_ids.append(self.vindex_ch[i]) # variable indices for coupling
                    self.c_ids.append(self.cindex_ch[i]) # coupling quantity indices for coupling
                    self.cname.append('V_'+chn)
                    self.switch_V[i], self.vname[i] = 1, 'p_'+chn
                elif self.cq[i] == 'flux':
                    assert(self.vq[i]=='pressure')
                    self.cname.append('Q_'+chn)
                    self.switch_V[i], self.vname[i] = 0, 'p_'+chn
                    self.v_ids.append(self.vindex_ch[i]) # variable indices for coupling
                    self.c_ids.append(self.cindex_ch[i]) # coupling quantity indices for coupling
                elif self.cq[i] == 'pressure':
                    if self.vq[i] == 'volume':
                        self.switch_V[i], self.vname[i] = 1, 'V_'+chn
                    elif self.vq[i] == 'flux':
                        self.switch_V[i], self.vname[i] = 0, 'Q_'+chn
                    else:
                        raise ValueError("Variable quantity has to be volume or flux!")
                    self.cname.append('p_'+chn)
                    self.si[i] = 1 # switch indices of pressure / outflux
                    self.v_ids.append(self.vindex_ch[i]-self.si[i]) # variable indices for coupling
                    self.c_ids.append(self.off_io)
                    self.off_io+=1
                else:
                    raise NameError("Unknown coupling quantity!")

            elif self.chmodels[ch]['type']=='3D_fluid':
                assert(self.cq[i] == 'pressure')
                self.switch_V[i], self.vname[i] = 0, 'Q_'+chn
                if ch != 'ao': self.si[i] = 1 # switch indices of pressure / outflux
                # add inflow pressures to coupling name prefixes
                for m in range(self.chmodels[ch]['num_inflows']):
                    self.cname.append('p_'+chn+'_i'+str(m+1))
                    self.v_ids.append(ind_i[m])
                    self.c_ids.append(self.off_io)
                    self.off_io+=1
                # add outflow pressures to coupling name prefixes
                for m in range(self.chmodels[ch]['num_outflows']):
                    self.cname.append('p_'+chn+'_o'+str(m+1))
                    self.v_ids.append(ind_o[m])
                    self.c_ids.append(self.off_io)
                    self.off_io+=1
                # special case:
                # if we have an LV surrounded by 3D flow domains (LA and AO),
                # we have no (0D) in-/outflow and hence no coupling pressure;
                # but if we have a coronary circulation model, we need to pass an integrated
                # chamber pressure for its correct evaluation, hence we append a p_v_l_o1
                # to the coupling array
                if ch=='lv' and self.cormodel:
                    if self.chmodels['lv']['num_outflows']==0:
                        self.cname.append('p_v_l_o1')

            else:
                raise NameError("Unknown chamber model for chamber %s!" % (ch))


    # set coupling state (populate x and c vectors with Sympy symbols) according to case and coupling quantity (can be volume, flux, or pressure)
    def set_coupling_state(self, ch, chvars, chfncs=[]):

        if ch == 'lv': V_unstressed, i = self.V_v_l_u,  0
        if ch == 'rv': V_unstressed, i = self.V_v_r_u,  1
        if ch == 'la': V_unstressed, i = self.V_at_l_u, 2
        if ch == 'ra': V_unstressed, i = self.V_at_r_u, 3
        if ch == 'ao': V_unstressed, i = self.V_ar_sys_u, 4

        # "distributed" p variables
        num_pdist = len(chvars)-1

        # time-varying elastances
        if self.chmodels[ch]['type']=='0D_elast' or self.chmodels[ch]['type']=='0D_elast_prescr':
            chvars['VQ'] = chvars['pi1']/chfncs[0] + V_unstressed # V = p/E(t) + V_u
            self.fnc_.append(chfncs[0])

            # all "distributed" p are equal to "main" p of chamber (= pi1)
            for k in range(10): # no more than 10 distributed p's allowed
                if 'pi'+str(k+1) in chvars.keys(): chvars['pi'+str(k+1)] = chvars['pi1']
                if 'po'+str(k+1) in chvars.keys(): chvars['po'+str(k+1)] = chvars['pi1']

        # rigid
        elif self.chmodels[ch]['type']=='0D_rigid':
            chvars['VQ'] = 0

            # all "distributed" p are equal to "main" p of chamber (= pi1)
            for k in range(10): # no more than 10 distributed p's allowed
                if 'pi'+str(k+1) in chvars.keys(): chvars['pi'+str(k+1)] = chvars['pi1']
                if 'po'+str(k+1) in chvars.keys(): chvars['po'+str(k+1)] = chvars['pi1']

        # 3D solid mechanics model, or 0D prescribed volume/flux/pressure (non-primary variables!)
        elif self.chmodels[ch]['type']=='3D_solid' or self.chmodels[ch]['type']=='0D_prescr':

            # all "distributed" p are equal to "main" p of chamber (= pi1)
            for k in range(10): # no more than 10 distributed p's allowed
                if 'pi'+str(k+1) in chvars.keys(): chvars['pi'+str(k+1)] = chvars['pi1']
                if 'po'+str(k+1) in chvars.keys(): chvars['po'+str(k+1)] = chvars['pi1']

            if self.cq[i] == 'volume' or self.cq[i] == 'flux':
                self.c_.append(chvars['VQ']) # V or Q
            if self.cq[i] == 'pressure':
                self.x_[self.vindex_ch[i]-self.si[i]] = chvars['VQ'] # V or Q
                self.c_.append(chvars['pi1'])

        # 3D fluid mechanics model
        elif self.chmodels[ch]['type']=='3D_fluid': # also for 2D FEM models

            assert(self.cq[i] == 'pressure' and self.vq[i] == 'flux')

            self.x_[self.vindex_ch[i]-self.si[i]] = chvars['VQ'] # Q of chamber is now variable

            # all "distributed" p that are not coupled are set to first inflow p
            for k in range(self.chmodels[ch]['num_inflows'],10):
                if 'pi'+str(k+1) in chvars.keys(): chvars['pi'+str(k+1)] = chvars['pi1']

            # if no inflow is present, set to zero
            if self.chmodels[ch]['num_inflows']==0: chvars['pi1'] = sp.S.Zero

            # now add inflow pressures to coupling array
            for m in range(self.chmodels[ch]['num_inflows']):
                self.c_.append(chvars['pi'+str(m+1)])

            # all "distributed" p that are not coupled are set to first outflow p
            for k in range(self.chmodels[ch]['num_outflows'],10):
                if 'po'+str(k+1) in chvars.keys(): chvars['po'+str(k+1)] = chvars['po1']

            # if no outflow is present, set to zero - except for special case:
            # if we have an LV surrounded by 3D flow domains (LA and AO),
            # we have no (0D) in-/outflow and hence no coupling pressure;
            # but if we have a coronary circulation model, we need to pass an integrated
            # chamber pressure for its correct evaluation, hence we append a p_v_l_o1
            # to the coupling array
            if self.chmodels[ch]['num_outflows']==0:
                if ch=='lv' and self.cormodel:
                    self.c_.append(chvars['po1'])
                    self.c_ids = [x+1 for x in self.c_ids]
                else:
                    chvars['po1'] = sp.S.Zero

            # now add outflow pressures to coupling array
            for m in range(self.chmodels[ch]['num_outflows']):
                self.c_.append(chvars['po'+str(m+1)])

        else:
            raise NameError("Unknown chamber model for chamber %s!" % (ch))


    # evaluate time-dependent state of chamber (for 0D elastance models)
    def evaluate_chamber_state(self, y, t):

        chamber_funcs=[]

        ci=0
        for i, ch in enumerate(['lv','rv','la','ra']):

            if self.chmodels[ch]['type']=='0D_elast':

                if ch == 'lv': E_max, E_min = self.E_v_max_l,  self.E_v_min_l
                if ch == 'rv': E_max, E_min = self.E_v_max_r,  self.E_v_min_r
                if ch == 'la': E_max, E_min = self.E_at_max_l, self.E_at_min_l
                if ch == 'ra': E_max, E_min = self.E_at_max_r, self.E_at_min_r

                # time-varying elastance model (y should be normalized activation function provided by user)
                E_ch_t = (E_max - E_min) * y[i] + E_min

                chamber_funcs.append(E_ch_t)

            elif self.chmodels[ch]['type']=='0D_elast_prescr':

                E_ch_t = y[i]

                chamber_funcs.append(E_ch_t)

            else:

                pass

        return chamber_funcs


    # initialize Lagrange multipliers for monolithic Lagrange-type coupling
    def initialize_lm(self, var, iniparam):

        ci=0
        for ch in ['lv','rv','la','ra', 'ao']:

            # name mapping
            if ch == 'lv': chn = 'v_l'
            if ch == 'rv': chn = 'v_r'
            if ch == 'la': chn = 'at_l'
            if ch == 'ra': chn = 'at_r'
            if ch == 'ao': chn = 'aort_sys'

            if self.chmodels[ch]['type']=='3D_solid':

                if 'p_'+chn+'_0' in iniparam.keys(): var[ci] = iniparam['p_'+chn+'_0']
                ci+=1

            if self.chmodels[ch]['type']=='3D_fluid':

                # in-flow pressures
                for m in range(self.chmodels[ch]['num_inflows']):
                    if 'p_'+chn+'_i'+str(m+1) in iniparam.keys(): var[ci] = iniparam['p_'+chn+'_i'+str(m+1)]
                    ci+=1

                # out-flow pressures
                for m in range(self.chmodels[ch]['num_outflows']):
                    if 'p_'+chn+'_o'+str(m+1) in iniparam.keys(): var[ci] = iniparam['p_'+chn+'_o'+str(m+1)]
                    ci+=1
