#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from ufl import tr, det, dot, grad, inv, dev, inner, Identity, variable, ln, sqrt, exp, diff, conditional, ge, gt, outer, cross, as_tensor, indices, as_ufl, constantvalue
from petsc4py import PETSc

from solid_material import materiallaw, growth, growthfunction
from projection import project

# nonlinear finite strain kinematics and constitutive class

class constitutive:
    
    def __init__(self, kin, materials, incompr_2field, mat_growth=None, mat_remodel=None):
        
        self.kin = kin
        
        self.matmodels = []
        for i in range(len(materials.keys())):
            self.matmodels.append(list(materials.keys())[i])
        
        self.matparams = []
        for i in range(len(materials.values())):
            self.matparams.append(list(materials.values())[i])
        
        self.mat_growth = mat_growth
        self.mat_remodel = mat_remodel
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
        self.I = Identity(3)


    # 2nd Piola-Kirchhoff stress core routine
    # we have everything in a Total Lagrangian setting and use S and C to express our internal virtual work
    def S(self, u_, p_, ivar=None, rvar=None, tang=False):
        
        C_ = variable(self.kin.C(u_))

        stress = constantvalue.zero((3,3))

        # volumetric (kinematic) growth
        if self.mat_growth:

            theta_ = ivar["theta"]
            
            # material has to be evaluated with C_e only, however total S has
            # to be computed by differentiating w.r.t. C (S = 2*dPsi/dC)
            self.mat = materiallaw(self.C_e(C_,theta_),self.I)

        else:
            
            self.mat = materiallaw(C_,self.I)
        
        m = 0
        for matlaw in self.matmodels:

            stress += self.add_stress_mat(matlaw, self.matparams[m], ivar, rvar, C_)

            m += 1
        
        # add remodeled material
        if self.mat_growth and self.mat_remodel:
            
            self.stress_base = stress
            
            self.stress_remod = constantvalue.zero((3,3))
            
            m = 0
            for matlaw in self.matmodels_remod:
            
                self.stress_remod += self.add_stress_mat(matlaw, self.matparams_remod[m], ivar, rvar, C_)

                m += 1
            
            # update the stress expression: S = (1-phi(theta)) * S_base + phi(theta) * S_remod
            stress = (1.-self.phi_remod(theta_))*self.stress_base + self.phi_remod(theta_)*self.stress_remod

        # if we have p (hydr. pressure) as variable in a 2-field functional
        if self.incompr_2field:
            if self.mat_growth:
                # TeX: S_{\mathrm{vol}} = -2 \frac{\partial[p(J^{\mathrm{e}}-1)]}{\partial \boldsymbol{C}}
                stress += -2.*diff(p_*(sqrt(det(self.C_e(C_,theta_)))-1.),C_)     
            else:
                # TeX: S_{\mathrm{vol}} = -2 \frac{\partial[p(J-1)]}{\partial \boldsymbol{C}} = -Jp\boldsymbol{C}^{-1}
                stress += -2.*diff(p_*(sqrt(det(C_))-1.),C_)

        if tang:
            return 2.*diff(stress,C_)
        else:
            return stress


    # add stress contributions from materials
    def add_stress_mat(self, matlaw, mparams, ivar, rvar, C_):

        # sanity check
        if self.incompr_2field and '_vol' in matlaw:
            raise AttributeError("Do not use a volumetric material law when using a 2-field variational principle with pressure dofs!")

        if matlaw == 'neohooke_dev':
            
            return self.mat.neohooke_dev(mparams,C_)
            
        elif matlaw == 'mooneyrivlin_dev':
            
            return self.mat.mooneyrivlin_dev(mparams,C_)
        
        elif matlaw == 'exponential_dev':
            
            return self.mat.exponential_dev(mparams,C_)
            
        elif matlaw == 'holzapfelogden_dev':

            return self.mat.holzapfelogden_dev(mparams,self.kin.fib_funcs[0],self.kin.fib_funcs[1],C_)
        
        elif matlaw == 'guccione_dev':

            return self.mat.guccione_dev(mparams,self.kin.fib_funcs[0],self.kin.fib_funcs[1],C_)
            
        elif matlaw == 'stvenantkirchhoff':
            
            return self.mat.stvenantkirchhoff(mparams,C_)
            
        elif matlaw == 'ogden_vol':
            
            return self.mat.ogden_vol(mparams,C_)
            
        elif matlaw == 'sussmanbathe_vol':
            
            return self.mat.sussmanbathe_vol(mparams,C_)

        elif matlaw == 'visco':
            
            dEdt_ = rvar["dEdt"][0]
            return self.mat.visco(mparams,dEdt_)
            
        elif matlaw == 'active_fiber':
            
            tau_a_ = ivar["tau_a"]
            return self.mat.active_fiber(tau_a_, self.kin.fib_funcs[0])
            
        elif matlaw == 'active_iso':
            
            tau_a_ = ivar["tau_a"]
            return self.mat.active_iso(tau_a_)
            
        elif matlaw == 'inertia':
            # density is added to kinetic virtual work
            return constantvalue.zero((3,3))
        
        elif matlaw == 'rayleigh_damping':
            # Rayleigh damping is added to virtual work
            return constantvalue.zero((3,3))
        
        elif matlaw == 'growth':
            # growth (and remodeling) treated separately
            return constantvalue.zero((3,3))
            
        else:

            raise NameError('Unknown solid material law!')


    # Cauchy stress tensor: sigma = (1/J) * F*S*F.T
    def sigma(self, u_, p_, ivar, rvar):
        return (1./self.kin.J(u_)) * self.kin.F(u_)*self.S(u_,p_,ivar,rvar)*self.kin.F(u_).T
    
    
    # deviatoric part of Cauchy stress tensor: sigma_dev = sigma - tr(sigma)/3 I
    def sigma_dev(self, u_, p_, ivar, rvar):
        return dev(self.sigma(u_,p_,ivar,rvar))
    
    
    # von Mises Cauchy stress
    def sigma_vonmises(self, u_, p_, ivar, rvar):
        return sqrt(3.*0.5*inner(self.sigma_dev(u_,p_,ivar,rvar),self.sigma_dev(u_,p_,ivar,rvar)))
    
    
    # 1st Piola-Kirchhoff stress tensor: P = F*S
    def P(self, u_, p_, ivar, rvar):
        return self.kin.F(u_)*self.S(u_,p_,ivar,rvar)
    
    
    # Kirchhoff stress tensor: tau = J * sigma
    def tau_kirch(self, u_, p_, ivar, rvar):
        return self.kin.J(u_) * self.sigma(u_)
    
    
    # Mandel stress tensor: M = C*S
    def M(self, u_, p_, ivar, rvar):
        return self.kin.C(u_)*self.S(u_,p_,ivar,rvar)
    
    
    # elastic 2nd Piola-Kirchhoff stress tensor
    def S_e(self, u_, p_, ivar, rvar):
        theta_ = ivar["theta"]
        return self.F_g(theta_) * self.S(u_,p_,ivar,rvar) * self.F_g(theta_).T
    
    
    # elastic Mandel stress tensor: M = C*S
    def M_e(self, u_, p_, C_, ivar, rvar):
        theta_ = ivar["theta"]
        return self.C_e(C_,theta_)*self.S_e(u_,p_,ivar,rvar)


    # viscous material tangent, for simple Green-Lagrange strain rate-dependent material with pseudo potential Psi_v = 0.5 * eta * dEdt : dEdt
    def Cvisco(self, eta, dt):
        i, j, k, l = indices(4)
        IFOUR = as_tensor(0.5*(self.I[i,k]*self.I[j,l] + self.I[i,l]*self.I[j,k]),(i,j,k,l))
        return eta*IFOUR/dt
        

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
            return diff(defgrd_gr,theta_)
        else:
            return defgrd_gr


    # elastic right Cauchy-Green tensor
    def C_e(self, C_, theta_):
        return inv(self.F_g(theta_)) * C_ * inv(self.F_g(theta_)).T
    
    # elastic fiber stretch
    def fibstretch_e(self, C_, theta_, fib_):
        return sqrt(dot(dot(self.C_e(C_,theta_),fib_), fib_))

    # elastic determinant of deformation gradient
    def J_e(self, u_, theta_):
        return det(self.kin.F(u_)*inv(self.F_g(theta_)))

    # dJe/dC, Je is formulated in terms of C
    def dJedC(self, u_, theta_):
        C_ = variable(self.kin.C(u_))
        Je = sqrt(det(self.C_e(C_, theta_)))
        return diff(Je,C_)


    # growth velocity gradient tensor
    def L(self, theta_, theta_old, dt):
        return dot((self.F_g(theta_) - self.F_g(theta_old))/dt, inv(self.F_g(theta_)))


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
            return diff(phi,theta_)
        else:
            return phi


    # growth residual and increment at Gauss point
    def res_dtheta_growth(self, u_, p_, ivar, rvar, theta_old_, dt, grparfuncs, rquant):
        
        theta_ = ivar["theta"]
        
        grfnc = growthfunction(theta_,self.I)

        thres = grparfuncs['growth_thres']

        try: omega = grparfuncs['thres_tol']
        except: omega = 0
        
        try: reduc = grparfuncs['trigger_reduction']
        except: reduc = 1
        
        # threshold should not be lower than specified (only relevant for multiscale analysis, where threshold is set element-wise)
        threshold = conditional(gt(thres,grparfuncs['growth_thres_0']), (1.+omega)*thres, grparfuncs['growth_thres_0'])

        # trace of elastic Mandel stress
        if self.growth_trig == 'volstress':
            
            trigger = reduc * tr(self.M_e(u_,p_,self.kin.C(u_),ivar,rvar))
            
        # elastic fiber stretch
        elif self.growth_trig == 'fibstretch':
            
            trigger = reduc * self.fibstretch_e(self.kin.C(u_),theta_,self.kin.fib_funcs[0])

        else:
            raise NameError("Unknown growth_trig!")

        # growth function
        ktheta = grfnc.grfnc1(trigger, threshold, grparfuncs)
        
        # growth residual
        r_growth = theta_ - theta_old_ - ktheta * (trigger - threshold) * dt

        # tangent
        K_growth = diff(r_growth,theta_)

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


    
    def dtheta_dC(self, u_, p_, ivar, rvar, theta_old_, dt, grparfuncs):
        
        theta_ = ivar["theta"]
        
        dFg_dtheta = self.F_g(theta_,tang=True)
        
        ktheta = self.res_dtheta_growth(u_, p_, ivar, rvar, theta_old_, dt, grparfuncs, 'ktheta')
        K_growth = self.res_dtheta_growth(u_, p_, ivar, rvar, theta_old_, dt, grparfuncs, 'tang')
        
        i, j, k, l = indices(4)
        
        if self.growth_trig == 'volstress':
            
            Cmat = self.S(u_,p_,ivar,rvar,tang=True)
            
            # TeX: \frac{\partial \vartheta}{\partial \boldsymbol{C}} = \frac{k(\vartheta) \Delta t}{\frac{\partial r}{\partial \vartheta}}\left(\boldsymbol{S} + \boldsymbol{C} : \frac{1}{2} \check{\mathbb{C}}\right)

            tangdC = (ktheta*dt/K_growth) * (self.S(u_,p_,ivar,rvar) + 0.5*as_tensor(self.kin.C(u_)[i,j]*Cmat[i,j,k,l], (k,l)))
            
        elif self.growth_trig == 'fibstretch':
            
            # TeX: \frac{\partial \vartheta}{\partial \boldsymbol{C}} = \frac{k(\vartheta) \Delta t}{\frac{\partial r}{\partial \vartheta}} \frac{1}{2\lambda_{f}} \boldsymbol{f}_{0}\otimes \boldsymbol{f}_{0}

            tangdC = (ktheta*dt/K_growth) * outer(self.kin.fib_funcs[0],self.kin.fib_funcs[0])/(2.*self.kin.fibstretch(u_,self.kin.fib_funcs[0]))

        else:
            raise NameError("Unkown growth_trig!")
        
        return tangdC

    # with deformation-dependent growth, the full material tangent reads
    # TeX: \mathbb{C} = 2\frac{\partial\boldsymbol{S}}{\partial\boldsymbol{C}} + 2 \left(\frac{\partial\boldsymbol{S}}{\partial\boldsymbol{F}^{\mathrm{g}}} : \frac{\partial\boldsymbol{F}^{\mathrm{g}}}{\partial\vartheta}\right) \otimes \frac{\partial \vartheta}{\partial \boldsymbol{C}} = \check{\mathbb{C}} + \tilde{\mathbb{C}}

    # TeX: \frac{\partial\boldsymbol{S}}{\partial\boldsymbol{F}^{\mathrm{g}}} = -\left(\boldsymbol{F}^{\mathrm{g}^{-1}}\overline{\otimes} \boldsymbol{S} +  \boldsymbol{S}\underline{\otimes}\boldsymbol{F}^{\mathrm{g}^{-1}}\right) - \left(\boldsymbol{F}^{\mathrm{g}^{-1}}\overline{\otimes}\boldsymbol{F}^{\mathrm{g}^{-1}}\right):\frac{1}{2}\check{\mathbb{C}}^{\mathrm{e}} : \left(\boldsymbol{F}^{\mathrm{g}^{-\mathrm{T}}}\overline{\otimes} \boldsymbol{C}^{\mathrm{e}} +  \boldsymbol{C}^{\mathrm{e}}\underline{\otimes}\boldsymbol{F}^{\mathrm{g}^{-\mathrm{T}}}\right)
    # definitions of outer 2nd order tensor products:
    # \mathbb{I}             = \boldsymbol{1}\,\overline{\otimes}\,\boldsymbol{1} = \delta_{ik}\delta_{jl} \; \hat{\boldsymbol{e}}_{i} \otimes \hat{\boldsymbol{e}}_{j} \otimes \hat{\boldsymbol{e}}_{k} \otimes \hat{\boldsymbol{e}}_{l}
    # \bar{\mathbb{I}}       = \boldsymbol{1}\,\underline{\otimes}\,\boldsymbol{1} = \delta_{il}\delta_{jk} \; \hat{\boldsymbol{e}}_{i} \otimes \hat{\boldsymbol{e}}_{j} \otimes \hat{\boldsymbol{e}}_{k} \otimes \hat{\boldsymbol{e}}_{l}
    # \bar{\bar{\mathbb{I}}} = \boldsymbol{1}\otimes\boldsymbol{1} = \delta_{ij}\delta_{kl} \; \hat{\boldsymbol{e}}_{i} \otimes \hat{\boldsymbol{e}}_{j} \otimes \hat{\boldsymbol{e}}_{k} \otimes \hat{\boldsymbol{e}}_{l}
    def dS_dFg(self, u_, p_, ivar, rvar, theta_old_, dt):
        
        theta_ = ivar["theta"]

        Cmat = self.S(u_,p_,ivar,rvar,tang=True)

        i, j, k, l, m, n = indices(6)
        
        # elastic material tangent (living in intermediate growth configuration)
        Cmat_e = dot(self.F_g(theta_),dot(self.F_g(theta_),dot(Cmat, dot(self.F_g(theta_).T,self.F_g(theta_).T))))

        Fginv_outertop_S     = as_tensor(inv(self.F_g(theta_))[i,k]*self.S(u_,p_,ivar,rvar)[j,l], (i,j,k,l))
        S_outerbot_Fginv     = as_tensor(self.S(u_,p_,ivar,rvar)[i,l]*inv(self.F_g(theta_))[j,k], (i,j,k,l))
        Fginv_outertop_Fginv = as_tensor(inv(self.F_g(theta_))[i,k]*inv(self.F_g(theta_))[j,l], (i,j,k,l))
        FginvT_outertop_Ce   = as_tensor(inv(self.F_g(theta_)).T[i,k]*self.C_e(self.kin.C(u_),theta_)[j,l], (i,j,k,l))
        Ce_outerbot_FginvT   = as_tensor(self.C_e(self.kin.C(u_),theta_)[i,l]*inv(self.F_g(theta_)).T[j,k], (i,j,k,l))

        Cmat_e_with_C_e = 0.5*as_tensor(Cmat_e[i,j,m,n]*(FginvT_outertop_Ce[m,n,k,l] + Ce_outerbot_FginvT[m,n,k,l]), (i,j,k,l))

        Fginv_outertop_Fginv_with_Cmat_e_with_C_e = as_tensor(Fginv_outertop_Fginv[i,j,m,n]*Cmat_e_with_C_e[m,n,k,l], (i,j,k,l))

        return -(Fginv_outertop_S + S_outerbot_Fginv) - Fginv_outertop_Fginv_with_Cmat_e_with_C_e



    # growth material tangent: Cgrowth = 2 (dS/dF_g : dF_g/dtheta) \otimes dtheta/dC
    # has to be set analytically, since nonlinear Gauss point theta cannot be expressed as
    # function of u, so ufl cannot take care of it...
    def Cgrowth(self, u_, p_, ivar, rvar, theta_old_, dt, grparfuncs):
        
        theta_ = ivar["theta"]
        
        dFg_dtheta = self.F_g(theta_,tang=True)
        
        i, j, k, l = indices(4)
        
        dtheta_dC_ = self.dtheta_dC(u_, p_, ivar, rvar, theta_old_, dt, grparfuncs)
        
        dS_dFg_ = self.dS_dFg(u_, p_, ivar, rvar, theta_old_, dt)

        dS_dFg_times_dFg_dtheta = as_tensor(dS_dFg_[i,j,k,l]*dFg_dtheta[k,l], (i,j))
        
        Cgrowth = 2.*as_tensor(dS_dFg_times_dFg_dtheta[i,j]*dtheta_dC_[k,l], (i,j,k,l))
        
        return Cgrowth


    # for a 2-field functional with u and p as variables, theta can depend on p in case of stress-mediated growth!
    def dtheta_dp(self, u_, p_, ivar, rvar, theta_old_, dt, grparfuncs):
        
        theta_ = ivar["theta"]
        
        dFg_dtheta = self.F_g(theta_,tang=True)
        
        ktheta = self.res_dtheta_growth(u_, p_, ivar, rvar, theta_old_, dt, grparfuncs, 'ktheta')
        K_growth = self.res_dtheta_growth(u_, p_, ivar, rvar, theta_old_, dt, grparfuncs, 'tang')
        
        if self.growth_trig == 'volstress':
            
            tangdp = (ktheta*dt/K_growth) * ( diff(tr(self.M_e(u_,p_,self.kin.C(u_),ivar,rvar)),p_) )
            
        elif self.growth_trig == 'fibstretch':
            
            tangdp = as_ufl(0)

        else:
            raise NameError("Unkown growth_trig!")
        
        return tangdp


    # growth material tangent for 2-field functional: Cgrowth_p = (dS/dF_g : dF_g/dtheta) * dtheta/dp
    # has to be set analytically, since nonlinear Gauss point theta cannot be expressed as
    # function of u, so ufl cannot take care of it...
    def Cgrowth_p(self, u_, p_, ivar, rvar, theta_old_, dt, grparfuncs):
        
        theta_ = ivar["theta"]
        
        dFg_dtheta = self.F_g(theta_,tang=True)
        
        i, j, k, l = indices(4)
        
        dtheta_dp_ = self.dtheta_dp(u_, p_, ivar, rvar, theta_old_, dt, grparfuncs)
        
        dS_dFg_ = self.dS_dFg(u_, p_, ivar, rvar, theta_old_, dt)

        dS_dFg_times_dFg_dtheta = as_tensor(dS_dFg_[i,j,k,l]*dFg_dtheta[k,l], (i,j))
        
        Cgrowth_p = dS_dFg_times_dFg_dtheta * dtheta_dp_
        
        return Cgrowth_p


    # remodeling material tangent: Cremod = 2 dphi/dC * (S_remod - S_base) = 2 dphi/dtheta * dtheta/dC * (S_remod - S_base)
    # has to be set analytically, since nonlinear Gauss point theta cannot be expressed as
    # function of u, so ufl cannot take care of it...
    def Cremod(self, u_, p_, ivar, rvar, theta_old_, dt, grparfuncs):
        
        theta_ = ivar["theta"]
        
        i, j, k, l = indices(4)
        
        dphi_dtheta_ = self.phi_remod(theta_,tang=True)
        dtheta_dC_ = self.dtheta_dC(u_, p_, ivar, rvar, theta_old_, dt, grparfuncs)

        Cremod = 2.*dphi_dtheta_ * as_tensor(dtheta_dC_[i,j]*(self.stress_remod - self.stress_base)[k,l], (i,j,k,l))

        return Cremod


    # remodeling material tangent for 2-field functional: Cremod_p = dphi/dp * (S_remod - S_base) = 2 dphi/dtheta * dtheta/dp * (S_remod - S_base)
    # has to be set analytically, since nonlinear Gauss point theta cannot be expressed as
    # function of u, so ufl cannot take care of it...
    def Cremod_p(self, u_, p_, ivar, rvar, theta_old_, dt, grparfuncs):
    
        theta_ = ivar["theta"]
        
        dphi_dtheta_ = self.phi_remod(theta_,tang=True)
        dtheta_dp_ = self.dtheta_dp(u_, p_, ivar, rvar, theta_old_, dt, grparfuncs)

        Cremod_p = 2.*dphi_dtheta_ * dtheta_dp_ * (self.stress_remod - self.stress_base)

        return Cremod_p


class kinematics:
    
    def __init__(self, fib_funcs=None, F_hist=None):
        
        # fibers
        self.fib_funcs = fib_funcs
        
        # history deformation gradient (for prestressing)
        self.F_hist = F_hist
        
        # identity tensor
        self.I = Identity(3)


    # deformation gradient: F = I + du/dx0
    def F(self, u_):
        if self.F_hist is not None:
            return self.F_hist*(self.I + grad(u_))
        else:
            return self.I + grad(u_)


    # determinant of deformation gradient: J = det(F)
    def J(self, u_):
        return det(self.F(u_))


    # dJ/dC = J/2 * C^-1, J is formulated as sqrt(det(C))
    def dJdC(self, u_):
        C_ = variable(self.C(u_))
        J = sqrt(det(C_))
        return diff(J,C_)


    # right Cauchy-Green tensor: C = F.T * F
    def C(self, u_):
        return self.F(u_).T*self.F(u_)
    
    
    # left Cauchy-Green tensor: b = F * F.T
    def b(self, u_):
        return self.F(u_)*self.F(u_).T


    # Green-Lagrange strain tensor: E = 0.5*(C - I)
    def E(self, u_):
        return 0.5*(self.C(u_) - self.I)
    
    
    # Euler-Almansi strain tensor: e = 0.5*(I - b^-1)
    def e(self, u_):
        return 0.5*(self.I - inv(self.b(u_)))


    # fiber stretch
    def fibstretch(self, u_, fib_):
        return sqrt(dot(dot(self.C(u_),fib_), fib_))


    # prestressing update (MULF - Modified Updated Lagrangian Formulation, cf. Gee et al. 2010)
    def prestress_update(self, u_, V, dx_, u_pre=None):
        
        F_hist_proj = project(self.F(u_), V, dx_)
        self.F_hist.interpolate(F_hist_proj)
        
        # for springs
        if u_pre is not None:
            u_pre.vector.axpy(1.0, u_.vector)
            u_pre.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        u_.vector.set(0.0)
        u_.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
