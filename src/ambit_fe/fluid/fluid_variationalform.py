#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl
from ..variationalform import variationalform_base


# fluid mechanics variational forms class
# Principle of Virtual Power
# TeX: \delta \mathcal{P} = \delta \mathcal{P}_{\mathrm{kin}} + \delta \mathcal{P}_{\mathrm{int}} - \delta \mathcal{P}_{\mathrm{ext}} = 0, \quad \forall \; \delta\boldsymbol{v}
class variationalform(variationalform_base):

    def __init__(self, var_v, var_p, n, formulation='nonconservative'):
        self.var_v = var_v
        # self.var_p = var_p

        self.n = n
        self.formulation = formulation
        # for convenience, use naming n0 for ALE (it's a normal determined by the initial mesh!)
        self.n0 = self.n

    ### Kinetic virtual power \delta \mathcal{P}_{\mathrm{kin}}

    def deltaW_kin_navierstokes_transient(self, a, v, rho, ddomain, w=None, Fale=None):
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

    def deltaW_kin_navierstokes_steady(self, v, rho, ddomain, w=None, Fale=None):
        if self.formulation=='nonconservative':
            return rho*ufl.dot(ufl.grad(v) * v, self.var_v)*ddomain
        elif self.formulation=='conservative':
            return rho*ufl.dot(ufl.div(ufl.outer(v,v)), self.var_v)*ddomain
        else:
            raise ValueError("Unknown fluid formulation!")

    def deltaW_kin_stokes_transient(self, a, v, rho, ddomain, w=None, Fale=None):

        return rho*ufl.dot(a, self.var_v)*ddomain

    ### Internal virtual power \delta \mathcal{P}_{\mathrm{int}}

    # TeX: \int\limits_{\Omega}\boldsymbol{\sigma} : \boldsymbol{\nabla}(\delta\boldsymbol{v})\,\mathrm{d}v
    def deltaW_int(self, sig, ddomain, Fale=None):

        return ufl.inner(sig, ufl.grad(self.var_v))*ddomain

    # TeX: \int\limits_{\Omega}\boldsymbol{\nabla}\cdot\boldsymbol{v}\,\delta p\,\mathrm{d}v
    def deltaW_int_pres(self, v, var_p, ddomain, w=None, Fale=None):

        return ufl.div(v)*var_p*ddomain

    def res_v_strong_navierstokes_transient(self, a, v, rho, sig, w=None, Fale=None):
        if self.formulation=='nonconservative':
            return rho*(a + ufl.grad(v) * v) - ufl.div(sig)
        elif self.formulation=='conservative':
            return rho*(a + ufl.div(ufl.outer(v,v))) - ufl.div(sig)
        else:
            raise ValueError("Unknown fluid formulation!")

    def res_v_strong_navierstokes_steady(self, v, rho, sig, w=None, Fale=None):
        if self.formulation=='nonconservative':
            return rho*(ufl.grad(v) * v) - ufl.div(sig)
        elif self.formulation=='conservative':
            return rho*(ufl.div(ufl.outer(v,v))) - ufl.div(sig)
        else:
            raise ValueError("Unknown fluid formulation!")

    def res_v_strong_stokes_transient(self, a, v, rho, sig, w=None, Fale=None):

        return rho*a - ufl.div(sig)

    def res_v_strong_stokes_steady(self, rho, sig, Fale=None):

        return -ufl.div(sig)

    def res_p_strong(self, v, Fale=None):

        return ufl.div(v)


    ### External virtual power \delta \mathcal{P}_{\mathrm{ext}}

    # Neumann load (Cauchy traction)
    # TeX: \int\limits_{\Gamma} \hat{\boldsymbol{t}} \cdot \delta\boldsymbol{v} \,\mathrm{d}a
    def deltaW_ext_neumann_cur(self, func, dboundary, Fale=None):

        return ufl.dot(func, self.var_v)*dboundary

    # Neumann load in current normal direction (Cauchy traction) - coincides with reference normal in Eulerian fluid mechanics
    # TeX: \int\limits_{\Gamma} p\,\boldsymbol{n}\cdot\delta\boldsymbol{v}\,\mathrm{d}a
    def deltaW_ext_neumann_normal_cur(self, func, dboundary, Fale=None):

        return func*ufl.dot(self.n, self.var_v)*dboundary

    # Neumann load in reference normal direction - coincides with reference normal in Eulerian fluid mechanics
    # TeX: \int\limits_{\Gamma_0} p\,\boldsymbol{n}\cdot\delta\boldsymbol{v}\,\mathrm{d}A
    def deltaW_ext_neumann_normal_ref(self, func, dboundary):

        return func*ufl.dot(self.n, self.var_v)*dboundary

    # body force external virtual power
    # TeX: \int\limits_{\Omega} \hat{\boldsymbol{b}}\cdot\delta\boldsymbol{v}\,\mathrm{d}V
    def deltaW_ext_bodyforce(self, func, funcdir, ddomain, Fale=None):

        return func*ufl.dot(funcdir, self.var_v)*ddomain

    # stabilized Neumann BC - Esmaily Moghadam et al. 2011
    def deltaW_ext_stabilized_neumann_cur(self, v, par1, par2, dboundary, w=None, Fale=None):

        vn = ufl.dot(v,self.n)
        return par1*((vn**2.)/(vn**2. + 0.01*par2**2.) * ufl.min_value(vn,0.) * ufl.dot(v,self.var_v)*dboundary) # version from Esmaily Moghadam et al. 2011 if par2 = 0

    # Robin condition (dashpot)
    # TeX: \int\limits_{\Gamma} c\,\boldsymbol{v}\cdot\delta\boldsymbol{v}\,\mathrm{d}a
    def deltaW_ext_robin_dashpot(self, v, c, dboundary, Fale=None):

        return -c*(ufl.dot(v, self.var_v)*dboundary)

    # Robin condition for valve, over internal surface
    def deltaW_ext_robin_valve(self, v, beta, dboundary, fcts='+', w=None, Fale=None):

        return (-(beta*ufl.dot((v-w), self.var_v)))(fcts)*dboundary

    # Robin condition (dashpot) in normal direction
    # TeX: \int\limits_{\Gamma} c\,(\boldsymbol{n}\otimes \boldsymbol{n})\boldsymbol{v}\cdot\delta\boldsymbol{v}\,\mathrm{d}a
    def deltaW_ext_robin_dashpot_normal_cur(self, v, c_n, dboundary, Fale=None):

        return -c_n*(ufl.dot(ufl.outer(self.n,self.n)*v, self.var_v)*dboundary)

    def deltaW_ext_robin_dashpot_normal_cross(self, v, c_c, dboundary, Fale=None):
        I = ufl.Identity(len(v))
        return -c_c*(ufl.dot((I - ufl.outer(self.n,self.n))*v, self.var_v)*dboundary)


    ### SUPG/PSPG stabilization - cf. Tezduyar and Osawa (2000), "Finite element stabilization parameters computed from element matrices and vectors"
    def stab_supg(self, a, v, p, res_v_strong, tau_supg, rho, ddomain, w=None, Fale=None):

        return (1./rho) * ufl.dot(tau_supg*rho*ufl.grad(self.var_v)*v, res_v_strong) * ddomain

    def stab_pspg(self, a, v, p, var_p, res_v_strong, tau_pspg, rho, ddomain, Fale=None):

        return (1./rho) * ufl.dot(tau_pspg*ufl.grad(var_p), res_v_strong) * ddomain

    def stab_lsic(self, v, tau_lsic, rho, ddomain, Fale=None):

        return tau_lsic*ufl.div(self.var_v)*rho*self.res_p_strong(v, Fale=Fale) * ddomain


    # reduced stabilization scheme - cf. Hoffman and Johnson (2006), "A new approach to computational turbulence modeling"
    def stab_v(self, delta1, delta2, delta3, v, p, ddomain, w=None, Fale=None, symmetric=False):

        if symmetric: # modification to make the effective stress symmetric
            return ( delta1 * ufl.dot(ufl.grad(v)*v, ufl.sym(ufl.grad(self.var_v))*v) + \
                     delta2 * ufl.div(v)*ufl.div(self.var_v) + \
                     delta3 * ufl.dot(ufl.grad(p), ufl.sym(ufl.grad(self.var_v))*v) ) * ddomain
        else:
            return ( delta1 * ufl.dot(ufl.grad(v)*v, ufl.grad(self.var_v)*v) + \
                     delta2 * ufl.div(v)*ufl.div(self.var_v) + \
                     delta3 * ufl.dot(ufl.grad(p), ufl.grad(self.var_v)*v) ) * ddomain

    def stab_p(self, delta1, delta3, v, p, var_p, rho, ddomain, w=None, Fale=None):

        return (1./rho) * ( delta1 * ufl.dot(ufl.grad(v)*v, ufl.grad(var_p)) + \
                            delta3 * ufl.dot(ufl.grad(p), ufl.grad(var_p)) ) * ddomain

    ### Flux coupling conditions

    # flux
    # TeX: \int\limits_{\Gamma} \boldsymbol{n}\cdot\boldsymbol{v}\,\mathrm{d}a
    def flux(self, v, dboundary, w=None, Fale=None, fcts=None):
        if fcts is None:
            return ufl.dot(self.n, v)*dboundary
        else:
            return (ufl.dot(self.n, v))(fcts)*dboundary



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

    def deltaW_kin_navierstokes_transient(self, a, v, rho, ddomain, w=None, Fale=None):
        J = ufl.det(Fale)

        if self.formulation=='nonconservative':
            # non-conservative form for ALE Navier-Stokes:
            # TeX:
            # \int\limits_{\Omega} \rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + (\boldsymbol{\nabla}\boldsymbol{v})(\boldsymbol{v}-\boldsymbol{w})\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}v =
            # \int\limits_{\Omega_0} J\rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + (\boldsymbol{\nabla}_{0}\boldsymbol{v}\,\boldsymbol{F}^{-1})(\boldsymbol{v}-\boldsymbol{w})\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}V
            return rho*ufl.dot(a + ufl.grad(v)*ufl.inv(Fale) * (v - w), self.var_v) * J*ddomain

        elif self.formulation=='conservative':
            # conservative form for ALE Navier-Stokes
            # TeX:
            # \int\limits_{\Omega} \rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + \boldsymbol{\nabla}\cdot(\boldsymbol{v}\otimes(\boldsymbol{v}-\boldsymbol{w}))\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}v =
            # \int\limits_{\Omega_0} J\rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + \boldsymbol{\nabla}_{0}(\boldsymbol{v}\otimes(\boldsymbol{v}-\boldsymbol{w})) : \boldsymbol{F}^{-\mathrm{T}}\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}V

            # note that we have div(v o (v-w)) = (grad v) (v-w) + v (div (v-w)) (Holzapfel eq. (1.292))
            # then use Holzapfel eq. (2.56)
            i, j, k = ufl.indices(3)
            return rho*ufl.dot(a + ufl.as_vector(ufl.grad(ufl.outer(v,v-w))[i,j,k]*ufl.inv(Fale).T[j,k], i), self.var_v) *J*ddomain

        else:
            raise ValueError("Unknown fluid formulation! Choose either 'nonconservative' or 'conservative'.")

    def deltaW_kin_navierstokes_steady(self, v, rho, ddomain, w=None, Fale=None):
        J = ufl.det(Fale)
        if self.formulation=='nonconservative':
            return rho*ufl.dot(ufl.grad(v)*ufl.inv(Fale) * (v - w), self.var_v) * J*ddomain
        elif self.formulation=='conservative':
            i, j, k = ufl.indices(3)
            return rho*ufl.dot(ufl.as_vector(ufl.grad(ufl.outer(v,v-w))[i,j,k]*ufl.inv(Fale).T[j,k], i), self.var_v) *J*ddomain
        else:
            raise ValueError("Unknown fluid formulation!")

    def deltaW_kin_stokes_transient(self, a, v, rho, ddomain, w=None, Fale=None):
        J = ufl.det(Fale)
        if self.formulation=='nonconservative':
            return rho*ufl.dot(a + ufl.grad(v)*ufl.inv(Fale) * (-w), self.var_v) * J*ddomain
        elif self.formulation=='conservative':
            i, j, k = ufl.indices(3)
            return rho*ufl.dot(a + ufl.as_vector(ufl.grad(ufl.outer(v,-w))[i,j,k]*ufl.inv(Fale).T[j,k], i), self.var_v) *J*ddomain
        else:
            raise ValueError("Unknown fluid formulation!")

    ### Internal virtual power \delta \mathcal{P}_{\mathrm{int}}

    # TeX:
    # \int\limits_{\Omega}\boldsymbol{\sigma} : \boldsymbol{\nabla}(\delta\boldsymbol{v})\,\mathrm{d}v =
    # \int\limits_{\Omega_0}J\boldsymbol{\sigma} : \boldsymbol{\nabla}_{0}(\delta\boldsymbol{v})\boldsymbol{F}^{-1}\,\mathrm{d}V (Holzapfel eq. (8.43))
    def deltaW_int(self, sig, ddomain, Fale=None):
        J = ufl.det(Fale)
        return ufl.inner(sig, ufl.grad(self.var_v)*ufl.inv(Fale)) * J*ddomain

    # TeX:
    # \int\limits_{\Omega}\boldsymbol{\nabla}\cdot\boldsymbol{v}\,\delta p\,\mathrm{d}v =
    # \int\limits_{\Omega_0}\boldsymbol{\nabla}_0\cdot(J\boldsymbol{F}^{-1}\boldsymbol{v})\,\delta p\,\mathrm{d}V
    # \int\limits_{\Omega_0}J\,\boldsymbol{\nabla}_0\boldsymbol{v} : \boldsymbol{F}^{-\mathrm{T}}\,\delta p\,\mathrm{d}V (cf. Holzapfel eq. (2.56))
    def deltaW_int_pres(self, v, var_p, ddomain, Fale=None):
        J = ufl.det(Fale)
        return ufl.inner(ufl.grad(v), ufl.inv(Fale).T)*var_p * J*ddomain

    # Robin term for weak imposition of Dirichlet condition
    # TeX:
    # \int\limits_{\Gamma} \beta\,(\boldsymbol{v}-\boldsymbol{v}_{\mathrm{D}})\cdot\delta\boldsymbol{v}\,\mathrm{d}a =
    # \int\limits_{\Gamma_0} J\beta\,(\boldsymbol{v}-\boldsymbol{v}_{\mathrm{D}})\cdot\delta\boldsymbol{v}\sqrt{\boldsymbol{n}_0 \cdot (\boldsymbol{F}^{-1}\boldsymbol{F}^{-\mathrm{T}})\boldsymbol{n}_0}\,\mathrm{d}A
    def deltaW_int_robin_cur(self, v, vD, beta, dboundary, Fale=None, fcts=None):
        J = ufl.det(Fale)
        if fcts is None:
            return beta*ufl.dot((v-vD), self.var_v) * J*ufl.sqrt(ufl.dot(self.n0, (ufl.inv(Fale)*ufl.inv(Fale).T)*self.n0))*dboundary
        else:
            return (beta*ufl.dot((v-vD), self.var_v) * J*ufl.sqrt(ufl.dot(self.n0, (ufl.inv(Fale)*ufl.inv(Fale).T)*self.n0)))(fcts)*dboundary

    def res_v_strong_navierstokes_transient(self, a, v, rho, sig, w=None, Fale=None):
        i, j, k = ufl.indices(3)
        if self.formulation=='nonconservative':
            return rho*(a + ufl.grad(v)*ufl.inv(Fale) * (v-w)) - ufl.as_vector(ufl.grad(sig)[i,j,k]*ufl.inv(Fale).T[j,k], i)
        elif self.formulation=='conservative':
            return rho*(a + ufl.as_vector(ufl.grad(ufl.outer(v,v-w))[i,j,k]*ufl.inv(Fale).T[j,k], i)) - ufl.as_vector(ufl.grad(sig)[l,m,n]*ufl.inv(Fale).T[j,k], i)
        else:
            raise ValueError("Unknown fluid formulation!")

    def res_v_strong_navierstokes_steady(self, v, rho, sig, w=None, Fale=None):
        i, j, k = ufl.indices(3)
        if self.formulation=='nonconservative':
            return rho*(ufl.grad(v)*ufl.inv(Fale) * (v-w)) - ufl.as_vector(ufl.grad(sig)[i,j,k]*ufl.inv(Fale).T[j,k], i)
        elif self.formulation=='conservative':
            return rho*(ufl.as_vector(ufl.grad(ufl.outer(v,v-w))[i,j,k]*ufl.inv(Fale).T[j,k], i)) - ufl.as_vector(ufl.grad(sig)[l,m,n]*ufl.inv(Fale).T[j,k], i)
        else:
            raise ValueError("Unknown fluid formulation!")

    def res_v_strong_stokes_transient(self, a, v, rho, sig, w=None, Fale=None):
        i, j, k = ufl.indices(3)
        if self.formulation=='nonconservative':
            return rho*(a + ufl.grad(v)*ufl.inv(Fale) * (-w)) - ufl.as_vector(ufl.grad(sig)[i,j,k]*ufl.inv(Fale).T[j,k], i)
        elif self.formulation=='conservative':
            return rho*(a + ufl.as_vector(ufl.grad(ufl.outer(v,-w))[i,j,k]*ufl.inv(Fale).T[j,k], i)) - ufl.as_vector(ufl.grad(sig)[i,j,k]*ufl.inv(Fale).T[j,k], i)
        else:
            raise ValueError("Unknown fluid formulation!")

    def res_v_strong_stokes_steady(self, rho, sig, Fale=None):
        i, j, k = ufl.indices(3)
        return -ufl.as_vector(ufl.grad(sig)[i,j,k]*ufl.inv(Fale).T[j,k], i)

    def res_p_strong(self, v, Fale=None):

        return ufl.inner(ufl.grad(v), ufl.inv(Fale).T)


    ### External virtual power \delta \mathcal{P}_{\mathrm{ext}}

    # Neumann load (Cauchy traction)
    # TeX:
    # \int\limits_{\Gamma} \hat{\boldsymbol{t}} \cdot \delta\boldsymbol{v} \,\mathrm{d}a =
    # \int\limits_{\Gamma_0} J\boldsymbol{F}^{-\mathrm{T}}\,\hat{\boldsymbol{t}} \cdot \delta\boldsymbol{v} \,\mathrm{d}A
    def deltaW_ext_neumann_cur(self, func, dboundary, Fale=None):
        J = ufl.det(Fale)
        return J*ufl.dot(ufl.inv(Fale).T*func, self.var_v)*dboundary

    # Neumann load in current normal direction (Cauchy traction)
    # TeX: \int\limits_{\Gamma} p\,\boldsymbol{n}\cdot\delta\boldsymbol{v}\,\mathrm{d}a =
    #      \int\limits_{\Gamma_0} p\,J\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_0\cdot\delta\boldsymbol{v}\,\mathrm{d}A
    def deltaW_ext_neumann_normal_cur(self, func, dboundary, Fale=None):
        J = ufl.det(Fale)
        return func*J*ufl.dot(ufl.inv(Fale).T*self.n0, self.var_v)*dboundary

    def deltaW_ext_neumann_ref(self, func, dboundary):

        return ufl.dot(func, self.var_v)*dboundary

    # body force external virtual power
    # TeX: \int\limits_{\Omega_0} J \hat{\boldsymbol{b}}\cdot\delta\boldsymbol{v}\,\mathrm{d}V
    def deltaW_ext_bodyforce(self, func, funcdir, ddomain, Fale=None):
        J = ufl.det(Fale)
        return func*ufl.dot(funcdir, self.var_v) *J*ddomain

    # stabilized Neumann BC - Esmaily Moghadam et al. 2011
    def deltaW_ext_stabilized_neumann_cur(self, v, par1, par2, dboundary, w=None, Fale=None):
        J = ufl.det(Fale)
        vwn = ufl.dot(v-w, J*ufl.inv(Fale).T*self.n)
        return par1*((vwn**2.)/(vwn**2. + 0.01*par2**2.) * ufl.min_value(vwn,0.) * ufl.dot(v,self.var_v)*dboundary) # version from Esmaily Moghadam et al. 2011 if param2 = 0

    # Robin condition (dashpot)
    # TeX:
    # \int\limits_{\Gamma} c\,\boldsymbol{v}\cdot\delta\boldsymbol{v}\,\mathrm{d}a =
    # \int\limits_{\Gamma_0} J c\,\boldsymbol{v}\cdot\delta\boldsymbol{v} \sqrt{\boldsymbol{n}_0 \cdot (\boldsymbol{F}^{-1}\boldsymbol{F}^{-\mathrm{T}})\boldsymbol{n}_0}\,\mathrm{d}A
    def deltaW_ext_robin_dashpot(self, v, c, dboundary, Fale=None):
        J = ufl.det(Fale)
        return -c*(ufl.dot(v, self.var_v) * J*ufl.sqrt(ufl.dot(self.n0, (ufl.inv(Fale)*ufl.inv(Fale).T)*self.n0))*dboundary)

    # Robin condition (dashpot) in normal direction
    def deltaW_ext_robin_dashpot_normal_cur(self, v, c_n, dboundary, Fale=None):
        raise ValueError("Robin condition in current normal direction not implemented")

    # Robin condition for valve, over internal surface
    def deltaW_ext_robin_valve(self, v, beta, dboundary, fcts='+', w=None, Fale=None):
        J = ufl.det(Fale)
        return (-(beta*ufl.dot((v-w), self.var_v) * J*ufl.sqrt(ufl.dot(self.n0, (ufl.inv(Fale)*ufl.inv(Fale).T)*self.n0))))(fcts)*dboundary


    ### SUPG/PSPG stabilization
    def stab_supg(self, a, v, p, res_v_strong, tau_supg, rho, ddomain, w=None, Fale=None):
        J = ufl.det(Fale)
        return (1./rho) * ufl.dot(tau_supg*rho*ufl.grad(self.var_v)*ufl.inv(Fale)*(v-w), res_v_strong) * J*ddomain

    def stab_pspg(self, a, v, p, var_p, res_v_strong, tau_pspg, rho, ddomain, Fale=None):
        J = ufl.det(Fale)
        return (1./rho) * ufl.dot(tau_pspg*ufl.inv(Fale).T*ufl.grad(var_p), res_v_strong) * J*ddomain

    def stab_lsic(self, v, tau_lsic, rho, ddomain, Fale=None):
        J = ufl.det(Fale)
        return tau_lsic * ufl.inner(ufl.grad(self.var_v),ufl.inv(Fale).T) * rho*self.res_p_strong(v, Fale=Fale) * J*ddomain

    def stab_v(self, delta1, delta2, delta3, v, p, ddomain, w=None, Fale=None, symmetric=False):
        J = ufl.det(Fale)
        if symmetric: # modification to make the effective stress symmetric
            return ( delta1 * ufl.dot(ufl.grad(v)*ufl.inv(Fale)*(v-w), ufl.sym(ufl.grad(self.var_v)*ufl.inv(Fale))*v) + \
                     delta2 * ufl.inner(ufl.grad(v),ufl.inv(Fale).T) * ufl.inner(ufl.grad(self.var_v),ufl.inv(Fale).T) + \
                     delta3 * ufl.dot(ufl.inv(Fale).T*ufl.grad(p), ufl.sym(ufl.grad(self.var_v)*ufl.inv(Fale))*v) ) * J*ddomain
        else:
            return ( delta1 * ufl.dot(ufl.grad(v)*ufl.inv(Fale)*(v-w), ufl.grad(self.var_v)*ufl.inv(Fale)*v) + \
                     delta2 * ufl.inner(ufl.grad(v),ufl.inv(Fale).T) * ufl.inner(ufl.grad(self.var_v),ufl.inv(Fale).T) + \
                     delta3 * ufl.dot(ufl.inv(Fale).T*ufl.grad(p), ufl.grad(self.var_v)*ufl.inv(Fale)*v) ) * J*ddomain

    def stab_p(self, delta1, delta3, v, p, var_p, rho, ddomain, w=None, Fale=None):
        J = ufl.det(Fale)
        return (1./rho) * ( delta1 * ufl.dot(ufl.grad(v)*ufl.inv(Fale)*(v-w), ufl.inv(Fale).T*ufl.grad(var_p)) + \
                            delta3 * ufl.dot(ufl.inv(Fale).T*ufl.grad(p), ufl.inv(Fale).T*ufl.grad(var_p)) ) * J*ddomain


    ### Flux coupling conditions

    # flux
    # TeX: \int\limits_{\Gamma} (\boldsymbol{v}-\boldsymbol{w})\cdot\boldsymbol{n}\,\mathrm{d}a =
    #      \int\limits_{\Gamma_0} (\boldsymbol{v}-\boldsymbol{w})\cdot J\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_0\,\mathrm{d}A
    def flux(self, v, dboundary, w=None, Fale=None, fcts=None):
        J = ufl.det(Fale)
        if fcts is None:
            return J*ufl.dot(ufl.inv(Fale).T*self.n0, (v-w))*dboundary
        else:
            return (J*ufl.dot(ufl.inv(Fale).T*self.n0, (v-w)))(fcts)*dboundary
