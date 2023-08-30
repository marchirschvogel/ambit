#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl
from petsc4py import PETSc

from .solid_material import materiallaw, growth, growthfunction


# nonlinear finite strain kinematics and constitutive class

class constitutive:

    def __init__(self, kin, materials, incompr_2field, mat_growth=None, mat_remodel=None, mat_plastic=None):

        self.kin = kin

        self.matmodels = []
        for i in range(len(materials.keys())):
            self.matmodels.append(list(materials.keys())[i])

        self.matparams = []
        for i in range(len(materials.values())):
            self.matparams.append(list(materials.values())[i])

        self.mat_growth = mat_growth
        self.mat_remodel = mat_remodel
        self.mat_plastic = mat_plastic
        self.incompr_2field = incompr_2field

        if self.mat_growth:
            # growth & remodeling parameters
            self.gandrparams = materials['growth']
            self.growth_dir = self.gandrparams['growth_dir']
            self.growth_trig = self.gandrparams['growth_trig']

            if self.mat_remodel:

                self.matmodels_remod = []
                for i in range(len(self.gandrparams['remodeling_mat'].keys())):
                    self.matmodels_remod.append(list(self.gandrparams['remodeling_mat'].keys())[i])

                self.matparams_remod = []
                for i in range(len(self.gandrparams['remodeling_mat'].values())):
                    self.matparams_remod.append(list(self.gandrparams['remodeling_mat'].values())[i])

        # identity tensor
        self.I = ufl.Identity(self.kin.dim)


    # 2nd Piola-Kirchhoff stress core routine
    # we have everything in a Total Lagrangian setting and use S and C to express our internal virtual work
    def S(self, u_, p_, v_, ivar=None, tang=False):

        C_ = ufl.variable(self.kin.C(u_))

        if not isinstance(v_, ufl.constantvalue.Zero):
            Cdot_ = ufl.variable(self.kin.Cdot(u_,v_))
        else:
            Cdot_ = ufl.constantvalue.zero((self.kin.dim,self.kin.dim))

        stress = ufl.constantvalue.zero((self.kin.dim,self.kin.dim))

        # volumetric (kinematic) growth
        if self.mat_growth:

            assert(not self.mat_plastic)

            theta_ = ivar["theta"]

            # material has to be evaluated with C_e (and Cdot_v) only, however total S has
            # to be computed by differentiating w.r.t. C (and Cdot)
            self.mat = materiallaw(self.C_e(C_,theta_), self.Cdot_v(Cdot_,theta_), self.I) # TODO: Using Cdot_v might not be consistent here, but C_edot...

        elif self.mat_plastic:

            assert(not self.mat_growth)

            raise ValueError("Plasticity not yet fully implemented!")

        else:

            self.mat = materiallaw(C_, Cdot_, self.I)

        m = 0
        for matlaw in self.matmodels:

            stress += self.add_stress_mat(matlaw, self.matparams[m], ivar, C_, Cdot_)

            m += 1

        # add remodeled material
        if self.mat_growth and self.mat_remodel:

            self.stress_base = stress

            self.stress_remod = ufl.constantvalue.zero((self.kin.dim,self.kin.dim))

            m = 0
            for matlaw in self.matmodels_remod:

                self.stress_remod += self.add_stress_mat(matlaw, self.matparams_remod[m], ivar, C_, Cdot_)

                m += 1

            # update the stress expression: S = (1-phi(theta)) * S_base + phi(theta) * S_remod
            stress = (1.-self.phi_remod(theta_))*self.stress_base + self.phi_remod(theta_)*self.stress_remod

        # if we have p (hydr. pressure) as variable in a 2-field functional
        if self.incompr_2field:
            if self.mat_growth:
                # TeX: S_{\mathrm{vol}} = -2 \frac{\partial[p(J^{\mathrm{e}}-1)]}{\partial \boldsymbol{C}}
                stress += -2.*ufl.diff(p_*(ufl.sqrt(ufl.det(self.C_e(C_,theta_)))-1.),C_)
            else:
                # TeX: S_{\mathrm{vol}} = -2 \frac{\partial[p(J-1)]}{\partial \boldsymbol{C}} = -Jp\boldsymbol{C}^{-1}
                stress += -2.*ufl.diff(p_*(ufl.sqrt(ufl.det(C_))-1.),C_)

        if tang:
            Cmat = 2.*ufl.diff(stress,C_)
            if not isinstance(v_, ufl.constantvalue.Zero):
                Cmat_v = 2.*ufl.diff(stress,Cdot_)
            else:
                Cmat_v = ufl.constantvalue.zero((self.kin.dim,self.kin.dim))
            return Cmat, Cmat_v
        else:
            return stress


    # add stress contributions from materials
    def add_stress_mat(self, matlaw, mparams, ivar, C_, Cdot_):

        # sanity check
        if self.incompr_2field and '_vol' in matlaw:
            raise AttributeError("Do not use a volumetric material law when using a 2-field variational principle with pressure dofs!")

        if matlaw == 'neohooke_dev':

            return self.mat.neohooke_dev(mparams,C_)

        elif matlaw == 'yeoh_dev':

            return self.mat.yeoh_dev(mparams,C_)

        elif matlaw == 'mooneyrivlin_dev':

            return self.mat.mooneyrivlin_dev(mparams,C_)

        elif matlaw == 'exponential_dev':

            return self.mat.exponential_dev(mparams,C_)

        elif matlaw == 'holzapfelogden_dev':

            return self.mat.holzapfelogden_dev(mparams,self.kin.fib_funcs[0],self.kin.fib_funcs[1],C_)

        elif matlaw == 'guccione_dev':

            return self.mat.guccione_dev(mparams,self.kin.fib_funcs[0],self.kin.fib_funcs[1],C_)

        elif matlaw == 'neohooke_compressible':

            return self.mat.neohooke_compressible(mparams,C_)

        elif matlaw == 'stvenantkirchhoff':

            return self.mat.stvenantkirchhoff(mparams,C_)

        elif matlaw == 'ogden_vol':

            return self.mat.ogden_vol(mparams,C_)

        elif matlaw == 'sussmanbathe_vol':

            return self.mat.sussmanbathe_vol(mparams,C_)

        elif matlaw == 'visco_green':

            return self.mat.visco_green(mparams,Cdot_)

        elif matlaw == 'active_fiber':

            tau_a_ = ivar["tau_a"]
            return self.mat.active_fiber(tau_a_,self.kin.fib_funcs[0])

        elif matlaw == 'active_iso':

            tau_a_ = ivar["tau_a"]
            return self.mat.active_iso(tau_a_)

        elif matlaw == 'inertia':
            # density is added to kinetic virtual work
            return ufl.constantvalue.zero((self.kin.dim,self.kin.dim))

        elif matlaw == 'growth':
            # growth (and remodeling) treated separately
            return ufl.constantvalue.zero((self.kin.dim,self.kin.dim))

        elif matlaw == 'plastic':
            # plasticity treated separately
            return ufl.constantvalue.zero((self.kin.dim,self.kin.dim))

        else:

            raise NameError('Unknown solid material law!')


    # Cauchy stress tensor: sigma = (1/J) * F*S*F^T
    def sigma(self, u_, p_, v_, ivar):
        return (1./self.kin.J(u_)) * self.kin.F(u_)*self.S(u_,p_,v_,ivar)*self.kin.F(u_).T


    # deviatoric part of Cauchy stress tensor: sigma_dev = sigma - tr(sigma)/3 I
    def sigma_dev(self, u_, p_, v_, ivar):
        return ufl.dev(self.sigma(u_,p_,v_,ivar))


    # von Mises Cauchy stress
    def sigma_vonmises(self, u_, p_, v_, ivar):
        return ufl.sqrt(3.*0.5*ufl.inner(self.sigma_dev(u_,p_,v_,ivar),self.sigma_dev(u_,p_,v_,ivar)))


    # 1st Piola-Kirchhoff stress tensor: P = F*S
    def P(self, u_, p_, v_, ivar):
        return self.kin.F(u_)*self.S(u_,p_,v_,ivar)


    # Kirchhoff stress tensor: tau = J * sigma
    def tau_kirch(self, u_, p_, v_, ivar):
        return self.kin.J(u_) * self.sigma(u_,p_,v_,ivar)


    # Mandel stress tensor: M = C*S
    def M(self, u_, p_, v_, ivar):
        return self.kin.C(u_)*self.S(u_,p_,v_,ivar)


    # elastic 2nd Piola-Kirchhoff stress tensor
    def S_e(self, u_, p_, v_, ivar):
        theta_ = ivar["theta"]
        return self.F_g(theta_) * self.S(u_,p_,v_,ivar) * self.F_g(theta_).T


    # elastic Mandel stress tensor: M = C*S
    def M_e(self, u_, p_, v_, C_, ivar):
        theta_ = ivar["theta"]
        return self.C_e(C_,theta_)*self.S_e(u_,p_,v_,ivar)


    # growth kinematics are here in the constitutive class, since this is initialized per material law
    # (we can have different mats with different growth settings, or some with and some without growth...),
    # while the kinematics class is once initialized for the whole problem

    # growth deformation gradient
    def F_g(self, theta_, tang=False):

        # split of deformation gradient into elastic and growth part: F = F_e*F_g
        gr = growth(theta_,self.I)

        if self.growth_dir == 'isotropic':
            defgrd_gr = gr.isotropic()
        elif self.growth_dir == 'fiber':
            defgrd_gr = gr.fiber(self.kin.fib_funcs[0])
        elif self.growth_dir == 'crossfiber':
            defgrd_gr = gr.crossfiber(self.kin.fib_funcs[0])
        elif self.growth_dir == 'radial':
            defgrd_gr = gr.radial(self.kin.fib_funcs[0],self.kin.fib_funcs[1])
        else:
            raise NameError("Unknown growth direction.")

        if tang:
            return ufl.diff(defgrd_gr,theta_)
        else:
            return defgrd_gr


    # elastic right Cauchy-Green tensor
    def C_e(self, C_, theta_):
        return ufl.inv(self.F_g(theta_)) * C_ * ufl.inv(self.F_g(theta_)).T

    # viscous right Cauchy-Green tensor
    def Cdot_v(self, Cdot_, theta_):
        return ufl.inv(self.F_g(theta_)) * Cdot_ * ufl.inv(self.F_g(theta_)).T

    # elastic fiber stretch
    def fibstretch_e(self, C_, theta_, fib_):
        return ufl.sqrt(ufl.dot(ufl.dot(self.C_e(C_,theta_),fib_), fib_))

    # elastic determinant of deformation gradient
    def J_e(self, u_, theta_):
        return ufl.det(self.kin.F(u_)*ufl.inv(self.F_g(theta_)))

    # dJe/dC, Je is formulated in terms of C
    def dJedC(self, u_, theta_):
        C_ = ufl.variable(self.kin.C(u_))
        Je = ufl.sqrt(ufl.det(self.C_e(C_, theta_)))
        return ufl.diff(Je,C_)

    # dF_g/dt
    def F_gdot(self, theta_, theta_old, dt):
        return (self.F_g(theta_) - self.F_g(theta_old))/dt

    # d(F_g^(-1))/dt = -F_g^(-1) dF_g/dt F_g^(-1) (cf. Holzapfel 2000, eq. (1.237))
    def F_gdot(self, theta_, theta_old, dt):
        return -ufl.inv(self.F_g(theta_)) * self.F_gdot(theta_, theta_old, dt) * ufl.inv(self.F_g(theta_))

    # growth velocity gradient tensor
    def L(self, theta_, theta_old, dt):
        return ufl.dot(self.F_gdot(theta_, theta_old, dt), ufl.inv(self.F_g(theta_)))

    # rate of elastic right Cauchy-Green tensor: note that d(F_g^(-T))/dt = [d(F_g^(-1))/dt]^T (cf. Holzapfel 2000, eq. (1.235))
    def C_edot(self, C_, Cdot_, theta_, theta_old, dt):
        return ufl.inv(self.F_gdot(theta_, theta_old, dt)) * C_ * ufl.inv(self.F_g(theta_)).T + ufl.inv(self.F_g(theta_)) * Cdot_ * ufl.inv(self.F_g(theta_)).T + ufl.inv(self.F_g(theta_)) * C_ * ufl.inv(self.F_gdot(theta_, theta_old, dt)).T


    # growth remodeling fraction: fraction of remodeled material
    # approach to include remodeling into volumetric growth constitutive models (similar to Thon et al. 2018, see Diss Hirschvogel p. 78ff.)
    # assumption is that the overall material composition after growth exhibits different passive or active mechanical properties
    # than prior to growth, depending on the fraction of grown to non-grown matter
    # assuming a cube that underwent isotropic, bi-axial, or uni-axial growth described by the growth stretch theta, the reference cube's
    # volume share in the grown cube can be stated as 1/theta^3, 1/theta^2, or 1/theta, respectively
    # so the fraction of the remodeled material is 1 - 1/theta^3, 1 - 1/theta^2, or 1 - 1/theta, respectively
    def phi_remod(self, theta_, tang=False):

        if self.growth_dir == 'isotropic': # tri-axial
            phi = 1.-1./(theta_*theta_*theta_)
        elif self.growth_dir == 'fiber': # uni-axial
            phi = 1.-1./theta_
        elif self.growth_dir == 'crossfiber': # bi-axial
            phi = 1.-1./(theta_*theta_)
        elif self.growth_dir == 'radial': # uni-axial
            phi = 1.-1./theta_
        else:
            raise NameError("Unknown growth direction.")

        if tang:
            return ufl.diff(phi,theta_)
        else:
            return phi


    # growth residual and increment at Gauss point
    def res_dtheta_growth(self, u_, p_, v_, ivar, theta_old_, dt, thres, rquant):

        theta_ = ivar["theta"]

        grfnc = growthfunction(theta_,self.I)

        thres = self.gandrparams['growth_thres']

        try: omega = self.gandrparams['thres_tol']
        except: omega = 0

        try: reduc = self.gandrparams['trigger_reduction']
        except: reduc = 1

        # threshold should not be lower than specified (only relevant for multiscale analysis, where threshold is set element-wise)
        threshold = ufl.conditional(ufl.gt(thres,self.gandrparams['growth_thres']), (1.+omega)*thres, self.gandrparams['growth_thres'])

        # trace of elastic Mandel stress
        if self.growth_trig == 'volstress':

            trigger = reduc * ufl.tr(self.M_e(u_,p_,v_,self.kin.C(u_),ivar))

        # elastic fiber stretch
        elif self.growth_trig == 'fibstretch':

            trigger = reduc * self.fibstretch_e(self.kin.C(u_),theta_,self.kin.fib_funcs[0])

        else:
            raise NameError("Unknown growth_trig!")

        # growth function
        ktheta = grfnc.grfnc1(trigger, threshold, self.gandrparams)

        # growth residual
        r_growth = theta_ - theta_old_ - ktheta * (trigger - threshold) * dt

        # tangent
        K_growth = ufl.diff(r_growth,theta_)

        # increment
        del_theta = -r_growth / K_growth

        if rquant=='res_del':
            return r_growth, del_theta
        elif rquant=='ktheta':
            return ktheta
        elif rquant=='tang':
            return K_growth
        else:
            raise NameError("Unknown return quantity!")


    def dtheta_dC(self, u_, p_, v_, ivar, theta_old_, dt, thres):

        theta_ = ivar["theta"]

        dFg_dtheta = self.F_g(theta_,tang=True)

        ktheta = self.res_dtheta_growth(u_, p_, v_, ivar, theta_old_, dt, thres, 'ktheta')
        K_growth = self.res_dtheta_growth(u_, p_, v_, ivar, theta_old_, dt, thres, 'tang')

        i, j, k, l = ufl.indices(4)

        if self.growth_trig == 'volstress':

            Cmat, _ = self.S(u_,p_,v_,ivar,tang=True)

            # TeX: \frac{\partial \vartheta}{\partial \boldsymbol{C}} = \frac{k(\vartheta) \Delta t}{\frac{\partial r}{\partial \vartheta}}\left(\boldsymbol{S} + \boldsymbol{C} : \frac{1}{2} \check{\mathbb{C}}\right)

            tangdC = (ktheta*dt/K_growth) * (self.S(u_,p_,v_,ivar) + 0.5*ufl.as_tensor(self.kin.C(u_)[i,j]*Cmat[i,j,k,l], (k,l)))

        elif self.growth_trig == 'fibstretch':

            # TeX: \frac{\partial \vartheta}{\partial \boldsymbol{C}} = \frac{k(\vartheta) \Delta t}{\frac{\partial r}{\partial \vartheta}} \frac{1}{2\lambda_{f}} \boldsymbol{f}_{0}\otimes \boldsymbol{f}_{0}

            tangdC = (ktheta*dt/K_growth) * ufl.outer(self.kin.fib_funcs[0],self.kin.fib_funcs[0])/(2.*self.kin.fibstretch(u_,self.kin.fib_funcs[0]))

        else:
            raise NameError("Unknown growth_trig!")

        return tangdC

    # with deformation-dependent growth, the full material tangent reads
    # TeX: \mathbb{C} = 2\frac{\partial\boldsymbol{S}}{\partial\boldsymbol{C}} + 2 \left(\frac{\partial\boldsymbol{S}}{\partial\boldsymbol{F}^{\mathrm{g}}} : \frac{\partial\boldsymbol{F}^{\mathrm{g}}}{\partial\vartheta}\right) \otimes \frac{\partial \vartheta}{\partial \boldsymbol{C}} = \check{\mathbb{C}} + \tilde{\mathbb{C}}

    # TeX: \frac{\partial\boldsymbol{S}}{\partial\boldsymbol{F}^{\mathrm{g}}} = -\left(\boldsymbol{F}^{\mathrm{g}^{-1}}\overline{\otimes} \boldsymbol{S} +  \boldsymbol{S}\underline{\otimes}\boldsymbol{F}^{\mathrm{g}^{-1}}\right) - \left(\boldsymbol{F}^{\mathrm{g}^{-1}}\overline{\otimes}\boldsymbol{F}^{\mathrm{g}^{-1}}\right):\frac{1}{2}\check{\mathbb{C}}^{\mathrm{e}} : \left(\boldsymbol{F}^{\mathrm{g}^{-\mathrm{T}}}\overline{\otimes} \boldsymbol{C}^{\mathrm{e}} +  \boldsymbol{C}^{\mathrm{e}}\underline{\otimes}\boldsymbol{F}^{\mathrm{g}^{-\mathrm{T}}}\right)
    # definitions of outer 2nd order tensor products:
    # \mathbb{I}             = \boldsymbol{1}\,\overline{\otimes}\,\boldsymbol{1} = \delta_{ik}\delta_{jl} \; \hat{\boldsymbol{e}}_{i} \otimes \hat{\boldsymbol{e}}_{j} \otimes \hat{\boldsymbol{e}}_{k} \otimes \hat{\boldsymbol{e}}_{l}
    # \bar{\mathbb{I}}       = \boldsymbol{1}\,\underline{\otimes}\,\boldsymbol{1} = \delta_{il}\delta_{jk} \; \hat{\boldsymbol{e}}_{i} \otimes \hat{\boldsymbol{e}}_{j} \otimes \hat{\boldsymbol{e}}_{k} \otimes \hat{\boldsymbol{e}}_{l}
    # \bar{\bar{\mathbb{I}}} = \boldsymbol{1}\otimes\boldsymbol{1} = \delta_{ij}\delta_{kl} \; \hat{\boldsymbol{e}}_{i} \otimes \hat{\boldsymbol{e}}_{j} \otimes \hat{\boldsymbol{e}}_{k} \otimes \hat{\boldsymbol{e}}_{l}
    def dS_dFg(self, u_, p_, v_, ivar, theta_old_, dt):

        theta_ = ivar["theta"]

        Cmat, _ = self.S(u_,p_,v_,ivar,tang=True)

        i, j, k, l, m, n = ufl.indices(6)

        # elastic material tangent (living in intermediate growth configuration)
        Cmat_e = ufl.dot(self.F_g(theta_),ufl.dot(self.F_g(theta_),ufl.dot(Cmat, ufl.dot(self.F_g(theta_).T,self.F_g(theta_).T))))

        Fginv_outertop_S     = ufl.as_tensor(ufl.inv(self.F_g(theta_))[i,k]*self.S(u_,p_,v_,ivar)[j,l], (i,j,k,l))
        S_outerbot_Fginv     = ufl.as_tensor(self.S(u_,p_,v_,ivar)[i,l]*ufl.inv(self.F_g(theta_))[j,k], (i,j,k,l))
        Fginv_outertop_Fginv = ufl.as_tensor(ufl.inv(self.F_g(theta_))[i,k]*ufl.inv(self.F_g(theta_))[j,l], (i,j,k,l))
        FginvT_outertop_Ce   = ufl.as_tensor(ufl.inv(self.F_g(theta_)).T[i,k]*self.C_e(self.kin.C(u_),theta_)[j,l], (i,j,k,l))
        Ce_outerbot_FginvT   = ufl.as_tensor(self.C_e(self.kin.C(u_),theta_)[i,l]*ufl.inv(self.F_g(theta_)).T[j,k], (i,j,k,l))

        Cmat_e_with_C_e = 0.5*ufl.as_tensor(Cmat_e[i,j,m,n]*(FginvT_outertop_Ce[m,n,k,l] + Ce_outerbot_FginvT[m,n,k,l]), (i,j,k,l))

        Fginv_outertop_Fginv_with_Cmat_e_with_C_e = ufl.as_tensor(Fginv_outertop_Fginv[i,j,m,n]*Cmat_e_with_C_e[m,n,k,l], (i,j,k,l))

        return -(Fginv_outertop_S + S_outerbot_Fginv) - Fginv_outertop_Fginv_with_Cmat_e_with_C_e


    # growth material tangent: Cgrowth = 2 (dS/dF_g : dF_g/dtheta) \otimes dtheta/dC
    # has to be set analytically, since nonlinear Gauss point theta cannot be expressed as
    # function of u, so ufl cannot take care of it...
    def Cgrowth(self, u_, p_, v_, ivar, theta_old_, dt, thres):

        theta_ = ivar["theta"]

        dFg_dtheta = self.F_g(theta_,tang=True)

        i, j, k, l = ufl.indices(4)

        dtheta_dC_ = self.dtheta_dC(u_, p_, v_, ivar, theta_old_, dt, thres)

        dS_dFg_ = self.dS_dFg(u_, p_, v_, ivar, theta_old_, dt)

        dS_dFg_times_dFg_dtheta = ufl.as_tensor(dS_dFg_[i,j,k,l]*dFg_dtheta[k,l], (i,j))

        Cgrowth = 2.*ufl.as_tensor(dS_dFg_times_dFg_dtheta[i,j]*dtheta_dC_[k,l], (i,j,k,l))

        return Cgrowth


    # for a 2-field functional with u and p as variables, theta can depend on p in case of stress-mediated growth!
    def dtheta_dp(self, u_, p_, v_, ivar, theta_old_, dt, thres):

        theta_ = ivar["theta"]

        dFg_dtheta = self.F_g(theta_,tang=True)

        ktheta = self.res_dtheta_growth(u_, p_, v_, ivar, theta_old_, dt, thres, 'ktheta')
        K_growth = self.res_dtheta_growth(u_, p_, v_, ivar, theta_old_, dt, thres, 'tang')

        if self.growth_trig == 'volstress':

            tangdp = (ktheta*dt/K_growth) * ( ufl.diff(ufl.tr(self.M_e(u_,p_,v_,self.kin.C(u_),ivar)),p_) )

        elif self.growth_trig == 'fibstretch':

            tangdp = as_ufl(0)

        else:
            raise NameError("Unknown growth_trig!")

        return tangdp


    # growth material tangent for 2-field functional: Cgrowth_p = (dS/dF_g : dF_g/dtheta) * dtheta/dp
    # has to be set analytically, since nonlinear Gauss point theta cannot be expressed as
    # function of u, so ufl cannot take care of it...
    def Cgrowth_p(self, u_, p_, v_, ivar, theta_old_, dt, thres):

        theta_ = ivar["theta"]

        dFg_dtheta = self.F_g(theta_,tang=True)

        i, j, k, l = ufl.indices(4)

        dtheta_dp_ = self.dtheta_dp(u_, p_, v_, ivar, theta_old_, dt, thres)

        dS_dFg_ = self.dS_dFg(u_, p_, v_, ivar, theta_old_, dt)

        dS_dFg_times_dFg_dtheta = ufl.as_tensor(dS_dFg_[i,j,k,l]*dFg_dtheta[k,l], (i,j))

        Cgrowth_p = dS_dFg_times_dFg_dtheta * dtheta_dp_

        return Cgrowth_p


    # remodeling material tangent: Cremod = 2 dphi/dC * (S_remod - S_base) = 2 dphi/dtheta * dtheta/dC * (S_remod - S_base)
    # has to be set analytically, since nonlinear Gauss point theta cannot be expressed as
    # function of u, so ufl cannot take care of it...
    def Cremod(self, u_, p_, v_, ivar, theta_old_, dt, thres):

        theta_ = ivar["theta"]

        i, j, k, l = ufl.indices(4)

        dphi_dtheta_ = self.phi_remod(theta_,tang=True)
        dtheta_dC_ = self.dtheta_dC(u_, p_, v_, ivar, theta_old_, dt, thres)

        Cremod = 2.*dphi_dtheta_ * ufl.as_tensor(dtheta_dC_[i,j]*(self.stress_remod - self.stress_base)[k,l], (i,j,k,l))

        return Cremod


    # remodeling material tangent for 2-field functional: Cremod_p = dphi/dp * (S_remod - S_base) = 2 dphi/dtheta * dtheta/dp * (S_remod - S_base)
    # has to be set analytically, since nonlinear Gauss point theta cannot be expressed as
    # function of u, so ufl cannot take care of it...
    def Cremod_p(self, u_, p_, v_, ivar, theta_old_, dt, thres):

        theta_ = ivar["theta"]

        dphi_dtheta_ = self.phi_remod(theta_,tang=True)
        dtheta_dp_ = self.dtheta_dp(u_, p_, v_, ivar, theta_old_, dt, thres)

        Cremod_p = 2.*dphi_dtheta_ * dtheta_dp_ * (self.stress_remod - self.stress_base)

        return Cremod_p



class kinematics:

    def __init__(self, dim, fib_funcs=None, u_pre=None):

        # physics dimension
        self.dim = dim

        # fibers
        self.fib_funcs = fib_funcs

        # prestress displacement
        self.u_pre = u_pre

        # identity tensor
        self.I = ufl.Identity(self.dim)


    # deformation gradient: F = I + du/dx0
    def F(self, u_, ext=False):
        if not ext:
            if self.u_pre is not None:
                return self.I + ufl.grad(u_+self.u_pre) # Schein and Gee 2021, equivalent to Gee et al. 2010
            else:
                return self.I + ufl.grad(u_)
        else:
            # prestress defgrad only enters internal force vector
            return self.I + ufl.grad(u_)


    # rate of deformation gradient: dF/dt = dv/dx0
    def Fdot(self, v_):

        if not isinstance(v_, ufl.constantvalue.Zero):
            return ufl.grad(v_)
        else:
            return ufl.constantvalue.zero((self.dim,self.dim))


    # determinant of deformation gradient: J = det(F)
    def J(self, u_, ext=False):
        return ufl.det(self.F(u_,ext))


    # dJ/dC = J/2 * C^-1, J is formulated as sqrt(det(C))
    def dJdC(self, u_):
        C_ = ufl.variable(self.C(u_))
        J = ufl.sqrt(ufl.det(C_))
        return ufl.diff(J,C_)


    # right Cauchy-Green tensor: C = F.T * F
    def C(self, u_):
        return self.F(u_).T*self.F(u_)


    # rate of right Cauchy-Green tensor: dC/dt = (dF/dt)^T F + F^T dF/dt
    def Cdot(self, u_, v_):
        return self.Fdot(v_).T*self.F(u_) + self.F(u_).T*self.Fdot(v_)


    # left Cauchy-Green tensor: b = F * F.T
    def b(self, u_):
        return self.F(u_)*self.F(u_).T


    # Green-Lagrange strain tensor: E = 0.5*(C - I)
    def E(self, u_):
        return 0.5*(self.C(u_) - self.I)


    # rate of Green-Lagrange strain tensor: dE/dt = 0.5 * dC/dt
    def Edot(self, u_, v_):
        return 0.5 * self.Cdot(u_,v_)


    # Euler-Almansi strain tensor: e = 0.5*(I - b^-1)
    def e(self, u_):
        return 0.5*(self.I - ufl.inv(self.b(u_)))


    # fiber stretch
    def fibstretch(self, u_, fib_):
        return ufl.sqrt(ufl.dot(ufl.dot(self.C(u_),fib_), fib_))


    # prestressing update (MULF - Modified Updated Lagrangian Formulation, cf. Gee et al. 2010,
    # displacement formulation according to Schein and Gee 2021)
    def prestress_update(self, u_):

        self.u_pre.vector.axpy(1.0, u_.vector)
        self.u_pre.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        u_.vector.set(0.0)
        u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
