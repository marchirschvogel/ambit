#!/usr/bin/env python3

# Copyright (c) 2019-2024, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl
from ..variationalform import variationalform_base

"""
Fluid mechanics variational forms class
Principle of Virtual Power
\delta \mathcal{P} = \delta \mathcal{P}_{\mathrm{kin}} + \delta \mathcal{P}_{\mathrm{int}} - \delta \mathcal{P}_{\mathrm{ext}} = 0, \quad \forall \; \delta\boldsymbol{v}
"""

class variationalform(variationalform_base):

    ### Kinetic virtual power \delta \mathcal{P}_{\mathrm{kin}}

    def deltaW_kin_navierstokes_transient(self, a, v, rho, ddomain, w=None, F=None):
        # standard Eulerian fluid
        if self.formulation=='nonconservative':
            # non-conservative form for Navier-Stokes:
            # TeX: \int\limits_{\Omega} \rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + (\boldsymbol{\nabla}\boldsymbol{v})\boldsymbol{v}\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}v
            return rho*ufl.dot(a + ufl.grad(v) * v, self.var_v)*ddomain

        elif self.formulation=='conservative':
            # conservative form for Navier-Stokes
            # TeX: \int\limits_{\Omega} \rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + \boldsymbol{\nabla}\cdot(\boldsymbol{v}\otimes\boldsymbol{v})\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}v

            # note that we have div(v o v) = (grad v) v + v (div v), where the latter term is the divergence condition (Holzapfel eq. (1.292))
            return rho*ufl.dot(a + ufl.div(ufl.outer(v,v)), self.var_v)*ddomain

        else:
            raise ValueError("Unknown fluid formulation! Choose either 'nonconservative' or 'conservative'.")

    def deltaW_kin_navierstokes_steady(self, v, rho, ddomain, w=None, F=None):
        if self.formulation=='nonconservative':
            return rho*ufl.dot(ufl.grad(v) * v, self.var_v)*ddomain
        elif self.formulation=='conservative':
            return rho*ufl.dot(ufl.div(ufl.outer(v,v)), self.var_v)*ddomain
        else:
            raise ValueError("Unknown fluid formulation!")

    def deltaW_kin_stokes_transient(self, a, v, rho, ddomain, w=None, F=None):

        return rho*ufl.dot(a, self.var_v)*ddomain

    ### Internal virtual power \delta \mathcal{P}_{\mathrm{int}}

    # TeX: \int\limits_{\Omega}\boldsymbol{\sigma} : \boldsymbol{\nabla}(\delta\boldsymbol{v})\,\mathrm{d}v
    def deltaW_int(self, sig, ddomain, F=None):

        return ufl.inner(sig, ufl.grad(self.var_v))*ddomain

    # TeX: \int\limits_{\Omega}\boldsymbol{\nabla}\cdot\boldsymbol{v}\,\delta p\,\mathrm{d}v
    def deltaW_int_pres(self, v, var_p, ddomain, w=None, F=None):

        return ufl.div(v)*var_p*ddomain

    def res_v_strong_navierstokes_transient(self, a, v, rho, sig, w=None, F=None):

        return self.f_inert_strong_navierstokes_transient(a, v, rho, w=w, F=F) - self.f_stress_strong(sig, F=F)

    def res_v_strong_navierstokes_steady(self, v, rho, sig, w=None, F=None):

        return self.f_inert_strong_navierstokes_steady(v, rho, w=w, F=F) - self.f_stress_strong(sig, F=F)

    def res_v_strong_stokes_transient(self, a, v, rho, sig, w=None, F=None):

        return self.f_inert_strong_stokes_transient(a, rho) - self.f_stress_strong(sig, F=F)

    def res_v_strong_stokes_steady(self, rho, sig, F=None):

        return -self.f_stress_strong(sig, F=F)

    def f_inert_strong_navierstokes_transient(self, a, v, rho, w=None, F=None):
        if self.formulation=='nonconservative':
            return rho*(a + ufl.grad(v) * v)
        elif self.formulation=='conservative':
            return rho*(a + ufl.div(ufl.outer(v,v)))
        else:
            raise ValueError("Unknown fluid formulation!")

    def f_inert_strong_navierstokes_steady(self, v, rho, w=None, F=None):
        if self.formulation=='nonconservative':
            return rho*(ufl.grad(v) * v)
        elif self.formulation=='conservative':
            return rho*(ufl.div(ufl.outer(v,v)))
        else:
            raise ValueError("Unknown fluid formulation!")

    def f_inert_strong_stokes_transient(self, a, rho):

        return rho*a

    def f_stress_strong(self, sig, F=None):

        return ufl.div(sig)

    def f_gradp_strong(self, p, F=None):

        return ufl.grad(p)

    def res_p_strong(self, v, F=None):

        return ufl.div(v)


    # stabilized Neumann BC - Esmaily Moghadam et al. 2011
    def deltaW_ext_stabilized_neumann(self, v, beta, dboundary, w=None, F=None):

        vn = ufl.dot(v,self.n0)
        return beta*(ufl.min_value(vn,0.) * ufl.dot(v,self.var_v)*dboundary)

    # mod. stabilized Neumann BC
    def deltaW_ext_stabilized_neumann_mod(self, v, beta, gamma, dboundary, w=None, F=None):

        vn = ufl.dot(v,self.n0)
        return beta*((vn**2.)/(vn**2. + 0.01*gamma**2.) * ufl.min_value(vn,0.) * ufl.dot(v,self.var_v)*dboundary) # version from Esmaily Moghadam et al. 2011 if gamma = 0


    # Robin condition for valve, over internal surface
    def deltaW_ext_robin_valve(self, v, beta, dboundary, fcts='+', w=None, F=None):

        return (-(beta*ufl.dot(v, self.var_v)))(fcts)*dboundary


    # Robin condition for valve, over internal surface - normal direction
    def deltaW_ext_robin_valve_normal_ref(self, v, beta, dboundary, fcts='+', w=None, F=None):

        return (-(beta*ufl.dot(ufl.outer(self.n0,self.n0)*v, self.var_v)))(fcts)*dboundary


    ### SUPG/PSPG stabilization - cf. Tezduyar and Osawa (2000), "Finite element stabilization parameters computed from element matrices and vectors"
    def stab_supg(self, v, res_v_strong, tau_supg, rho, ddomain, w=None, F=None, symmetric=False):

        if symmetric: # modification to make the effective stress symmetric - experimental, use with care...
            return (1./rho) * ufl.dot(tau_supg*rho*ufl.sym(ufl.grad(self.var_v))*v, res_v_strong) * ddomain
        else:
            return (1./rho) * ufl.dot(tau_supg*rho*ufl.grad(self.var_v)*v, res_v_strong) * ddomain

    def stab_pspg(self, var_p, res_v_strong, tau_pspg, rho, ddomain, F=None):

        return (1./rho) * ufl.dot(tau_pspg*ufl.grad(var_p), res_v_strong) * ddomain

    def stab_lsic(self, v, tau_lsic, rho, ddomain, F=None):

        return tau_lsic*ufl.div(self.var_v)*rho*self.res_p_strong(v, F=F) * ddomain


    # components of element-level Reynolds number - cf. Tezduyar and Osawa (2000)
    def re_c(self, rho, v, ddomain, w=None, F=None):

        return rho * ufl.dot(ufl.grad(v)*v, self.var_v) * ddomain

    def re_ktilde(self, rho, v, ddomain, w=None, F=None):

        return rho * ufl.dot(ufl.grad(v)*v, ufl.grad(self.var_v)*v) * ddomain


    ### Flux coupling conditions

    # flux
    # TeX: \int\limits_{\Gamma} \boldsymbol{n}\cdot\boldsymbol{v}\,\mathrm{d}a
    def flux(self, v, dboundary, w=None, F=None, fcts=None):
        if fcts is None:
            return ufl.dot(self.n0, v)*dboundary
        else:
            return (ufl.dot(self.n0, v))(fcts)*dboundary



# ALE fluid mechanics variational forms class (cf. https://w3.onera.fr/erc-aeroflex/project/strategies-for-coupling-the-fluid-and-solid-dynamics)
# Principle of Virtual Power
# TeX: \delta \mathcal{P} = \delta \mathcal{P}_{\mathrm{kin}} + \delta \mathcal{P}_{\mathrm{int}} - \delta \mathcal{P}_{\mathrm{ext}} = 0, \quad \forall \; \delta\boldsymbol{v}
# all infinitesimal volume elements transform according to
# \mathrm{d}v = J\,\mathrm{d}V
# a normal vector times surface element transforms according to Nanson's formula:
# \boldsymbol{n}\,\mathrm{d}a = J\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_0\,\mathrm{d}A
# hence, all infinitesimal surface elements transform according to
# \mathrm{d}a = J\sqrt{\boldsymbol{n}_0 \cdot (\boldsymbol{F}^{-1}\boldsymbol{F}^{-\mathrm{T}})\boldsymbol{n}_0}\,\mathrm{d}A

# gradients of a vector field transform according to:
# grad(u) = Grad(u) * F^(-1)
# gradients of a scalar field transform according to:
# grad(p) = F^(-T) * Grad(p)
# divergences of a tensor/vector field transform according to:
# div(A) = Grad(A) : F^(-T)

class variationalform_ale(variationalform):

    ### Kinetic virtual power \delta \mathcal{P}_{\mathrm{kin}}

    def deltaW_kin_navierstokes_transient(self, a, v, rho, ddomain, w=None, F=None):
        J = ufl.det(F)

        if self.formulation=='nonconservative':
            # non-conservative form for ALE Navier-Stokes:
            # TeX:
            # \int\limits_{\Omega} \rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + (\boldsymbol{\nabla}\boldsymbol{v})(\boldsymbol{v}-\boldsymbol{w})\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}v =
            # \int\limits_{\Omega_0} J\rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + (\boldsymbol{\nabla}_{0}\boldsymbol{v}\,\boldsymbol{F}^{-1})(\boldsymbol{v}-\boldsymbol{w})\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}V
            return rho*ufl.dot(a + ufl.grad(v)*ufl.inv(F) * (v - w), self.var_v) * J*ddomain

        elif self.formulation=='conservative':
            # conservative form for ALE Navier-Stokes
            # TeX:
            # \int\limits_{\Omega} \rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + \boldsymbol{\nabla}\cdot(\boldsymbol{v}\otimes(\boldsymbol{v}-\boldsymbol{w}))\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}v =
            # \int\limits_{\Omega_0} J\rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + \boldsymbol{\nabla}_{0}(\boldsymbol{v}\otimes(\boldsymbol{v}-\boldsymbol{w})) : \boldsymbol{F}^{-\mathrm{T}}\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}V

            # note that we have div(v o (v-w)) = (grad v) (v-w) + v (div (v-w)) (Holzapfel eq. (1.292))
            # then use Holzapfel eq. (2.56)
            i, j, k = ufl.indices(3)
            return rho*ufl.dot(a + ufl.as_vector(ufl.grad(ufl.outer(v,v-w))[i,j,k]*ufl.inv(F).T[j,k], i), self.var_v) *J*ddomain

        else:
            raise ValueError("Unknown fluid formulation! Choose either 'nonconservative' or 'conservative'.")

    def deltaW_kin_navierstokes_steady(self, v, rho, ddomain, w=None, F=None):
        J = ufl.det(F)
        if self.formulation=='nonconservative':
            return rho*ufl.dot(ufl.grad(v)*ufl.inv(F) * (v - w), self.var_v) * J*ddomain
        elif self.formulation=='conservative':
            i, j, k = ufl.indices(3)
            return rho*ufl.dot(ufl.as_vector(ufl.grad(ufl.outer(v,v-w))[i,j,k]*ufl.inv(F).T[j,k], i), self.var_v) *J*ddomain
        else:
            raise ValueError("Unknown fluid formulation!")

    def deltaW_kin_stokes_transient(self, a, v, rho, ddomain, w=None, F=None):
        J = ufl.det(F)
        if self.formulation=='nonconservative':
            return rho*ufl.dot(a + ufl.grad(v)*ufl.inv(F) * (-w), self.var_v) * J*ddomain
        elif self.formulation=='conservative':
            i, j, k = ufl.indices(3)
            return rho*ufl.dot(a + ufl.as_vector(ufl.grad(ufl.outer(v,-w))[i,j,k]*ufl.inv(F).T[j,k], i), self.var_v) *J*ddomain
        else:
            raise ValueError("Unknown fluid formulation!")

    ### Internal virtual power \delta \mathcal{P}_{\mathrm{int}}

    # TeX:
    # \int\limits_{\Omega}\boldsymbol{\sigma} : \boldsymbol{\nabla}(\delta\boldsymbol{v})\,\mathrm{d}v =
    # \int\limits_{\Omega_0}J\boldsymbol{\sigma} : \boldsymbol{\nabla}_{0}(\delta\boldsymbol{v})\boldsymbol{F}^{-1}\,\mathrm{d}V (Holzapfel eq. (8.43))
    def deltaW_int(self, sig, ddomain, F=None):
        J = ufl.det(F)
        return ufl.inner(sig, ufl.grad(self.var_v)*ufl.inv(F)) * J*ddomain

    # TeX:
    # \int\limits_{\Omega}\boldsymbol{\nabla}\cdot\boldsymbol{v}\,\delta p\,\mathrm{d}v =
    # \int\limits_{\Omega_0}\boldsymbol{\nabla}_0\cdot(J\boldsymbol{F}^{-1}\boldsymbol{v})\,\delta p\,\mathrm{d}V
    # \int\limits_{\Omega_0}J\,\boldsymbol{\nabla}_0\boldsymbol{v} : \boldsymbol{F}^{-\mathrm{T}}\,\delta p\,\mathrm{d}V (cf. Holzapfel eq. (2.56))
    def deltaW_int_pres(self, v, var_p, ddomain, F=None):
        J = ufl.det(F)
        return ufl.inner(ufl.grad(v), ufl.inv(F).T)*var_p * J*ddomain

    def res_v_strong_navierstokes_transient(self, a, v, rho, sig, w=None, F=None):

        return self.f_inert_strong_navierstokes_transient(a, v, rho, w=w, F=F) - self.f_stress_strong(sig, F=F)

    def res_v_strong_navierstokes_steady(self, v, rho, sig, w=None, F=None):

        return self.f_inert_strong_navierstokes_steady(v, rho, w=w, F=F) - self.f_stress_strong(sig, F=F)

    def res_v_strong_stokes_transient(self, a, v, rho, sig, w=None, F=None):

        return self.f_inert_strong_stokes_transient(a, v, rho, w=w, F=F) - self.f_stress_strong(sig, F=F)

    def res_v_strong_stokes_steady(self, rho, sig, F=None):

        return -self.f_stress_strong(sig, F=F)

    def f_inert_strong_navierstokes_transient(self, a, v, rho, w=None, F=None):
        if self.formulation=='nonconservative':
            return rho*(a + ufl.grad(v)*ufl.inv(F) * (v-w))
        elif self.formulation=='conservative':
            i, j, k = ufl.indices(3)
            return rho*(a + ufl.as_vector(ufl.grad(ufl.outer(v,v-w))[i,j,k]*ufl.inv(F).T[j,k], i))
        else:
            raise ValueError("Unknown fluid formulation!")

    def f_inert_strong_navierstokes_steady(self, v, rho, w=None, F=None):
        if self.formulation=='nonconservative':
            return rho*(ufl.grad(v)*ufl.inv(F) * (v-w))
        elif self.formulation=='conservative':
            i, j, k = ufl.indices(3)
            return rho*(ufl.as_vector(ufl.grad(ufl.outer(v,v-w))[i,j,k]*ufl.inv(F).T[j,k], i))
        else:
            raise ValueError("Unknown fluid formulation!")

    def f_inert_strong_stokes_transient(self, a, v, rho, w=None, F=None):
        if self.formulation=='nonconservative':
            return rho*(a + ufl.grad(v)*ufl.inv(F) * (-w))
        elif self.formulation=='conservative':
            i, j, k = ufl.indices(3)
            return rho*(a + ufl.as_vector(ufl.grad(ufl.outer(v,-w))[i,j,k]*ufl.inv(F).T[j,k], i))
        else:
            raise ValueError("Unknown fluid formulation!")

    def f_stress_strong(self, sig, F=None):
        i, j, k = ufl.indices(3)
        return ufl.as_vector(ufl.grad(sig)[i,j,k]*ufl.inv(F).T[j,k], i)

    def f_gradp_strong(self, p, F=None):

        return ufl.inv(F).T*ufl.grad(p)

    def res_p_strong(self, v, F=None):

        return ufl.inner(ufl.grad(v), ufl.inv(F).T)


    # stabilized Neumann BC - Esmaily Moghadam et al. 2011
    def deltaW_ext_stabilized_neumann(self, v, beta, dboundary, w=None, F=None):
        J = ufl.det(F)
        vwn = ufl.dot(v-w, J*ufl.inv(F).T*self.n0)
        return beta*(ufl.min_value(vwn,0.) * ufl.dot(v,self.var_v)*dboundary)

    # mod. stabilized Neumann BC
    def deltaW_ext_stabilized_neumann_mod(self, v, beta, gamma, dboundary, w=None, F=None):
        J = ufl.det(F)
        vwn = ufl.dot(v-w, J*ufl.inv(F).T*self.n0)
        return beta*((vwn**2.)/(vwn**2. + 0.01*gamma**2.) * ufl.min_value(vwn,0.) * ufl.dot(v,self.var_v)*dboundary) # version from Esmaily Moghadam et al. 2011 if gamma = 0


    # Robin condition for valve, over internal surface
    def deltaW_ext_robin_valve(self, v, beta, dboundary, fcts='+', w=None, F=None):

        return (-(beta*ufl.dot((v-w), self.var_v)))(fcts)*dboundary


    # Robin condition for valve, over internal surface - normal direction
    def deltaW_ext_robin_valve_normal_ref(self, v, beta, dboundary, fcts='+', w=None, F=None):

        return (-(beta*ufl.dot(ufl.outer(self.n0,self.n0)*(v-w), self.var_v)))(fcts)*dboundary


    ### SUPG/PSPG stabilization
    def stab_supg(self, v, res_v_strong, tau_supg, rho, ddomain, w=None, F=None, symmetric=False):
        J = ufl.det(F)
        if symmetric: # modification to make the effective stress symmetric - experimental, use with care...
            return (1./rho) * ufl.dot(tau_supg*rho*ufl.sym(ufl.grad(self.var_v)*ufl.inv(F))*v, res_v_strong) * J*ddomain
        else:
            return (1./rho) * ufl.dot(tau_supg*rho*ufl.grad(self.var_v)*ufl.inv(F)*v, res_v_strong) * J*ddomain

    def stab_pspg(self, var_p, res_v_strong, tau_pspg, rho, ddomain, F=None):
        J = ufl.det(F)
        return (1./rho) * ufl.dot(tau_pspg*ufl.inv(F).T*ufl.grad(var_p), res_v_strong) * J*ddomain

    def stab_lsic(self, v, tau_lsic, rho, ddomain, F=None):
        J = ufl.det(F)
        return tau_lsic * ufl.inner(ufl.grad(self.var_v),ufl.inv(F).T) * rho*self.res_p_strong(v, F=F) * J*ddomain


    # components of element-level Reynolds number
    def re_c(self, rho, v, ddomain, w=None, F=None):
        J = ufl.det(F)
        return rho * ufl.dot(ufl.grad(v)*ufl.inv(F)*(v-w), self.var_v) * J*ddomain

    def re_ktilde(self, rho, v, ddomain, w=None, F=None):
        J = ufl.det(F)
        return rho * ufl.dot(ufl.grad(v)*ufl.inv(F)*(v-w), ufl.grad(self.var_v)*ufl.inv(F)*v) * J*ddomain


    ### Flux coupling conditions

    # flux
    # TeX: \int\limits_{\Gamma} (\boldsymbol{v}-\boldsymbol{w})\cdot\boldsymbol{n}\,\mathrm{d}a =
    #      \int\limits_{\Gamma_0} (\boldsymbol{v}-\boldsymbol{w})\cdot J\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_0\,\mathrm{d}A
    def flux(self, v, dboundary, w=None, F=None, fcts=None):
        J = ufl.det(F)
        if fcts is None:
            return J*ufl.dot(ufl.inv(F).T*self.n0, (v-w))*dboundary
        else:
            return (J*ufl.dot(ufl.inv(F).T*self.n0, (v-w)))(fcts)*dboundary
