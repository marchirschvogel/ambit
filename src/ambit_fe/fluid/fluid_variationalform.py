#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
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
    # Kinetic virtual power \delta \mathcal{P}_{\mathrm{kin}}
    def deltaW_kin_navierstokes_transient(self, a, v, rho, ddomain, w=None, F=None, phi=None, phidot=None):
        rho_ = self.get_density(rho, phi=phi)
        # standard Eulerian fluid
        if self.formulation == "nonconservative":
            assert(phi is None)
            """ TeX:
            \int\limits_{\Omega} \rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + (\nabla\boldsymbol{v})\boldsymbol{v}\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}v
            """
            return rho_ * ufl.dot(a + ufl.grad(v) * v, self.var_v) * ddomain
        elif self.formulation == "conservative":
            if phi is not None:
                rhodot_ = ufl.diff(rho_,phi) * phidot
            else:
                rhodot_ = ufl.as_ufl(0)
            """ TeX:
            \int\limits_{\Omega} \left(\frac{\partial(\rho\boldsymbol{v})}{\partial t} + \nabla\cdot(\rho(\boldsymbol{v}\otimes\boldsymbol{v}))\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}v
            """
            return ufl.dot(rho_*a + rhodot_*v + ufl.div(rho_*ufl.outer(v, v)), self.var_v) * ddomain
        else:
            raise ValueError("Unknown fluid formulation! Choose either 'nonconservative' or 'conservative'.")

    def deltaW_kin_navierstokes_steady(self, v, rho, ddomain, F=None, phi=None, phidot=None):
        rho_ = self.get_density(rho, phi=phi)
        if self.formulation == "nonconservative":
            assert(phi is None)
            """ TeX:
            \int\limits_{\Omega} \rho (\nabla\boldsymbol{v})\boldsymbol{v} \cdot \delta\boldsymbol{v} \,\mathrm{d}v
            """
            return rho_ * ufl.dot(ufl.grad(v) * v, self.var_v) * ddomain
        elif self.formulation == "conservative":
            """ TeX:
            \int\limits_{\Omega} \nabla\cdot(\rho(\boldsymbol{v}\otimes\boldsymbol{v})) \cdot \delta\boldsymbol{v} \,\mathrm{d}v
            """
            return ufl.dot(ufl.div(rho_*ufl.outer(v, v)), self.var_v) * ddomain
        else:
            raise ValueError("Unknown fluid formulation!")

    def deltaW_kin_stokes_transient(self, a, v, rho, ddomain, w=None, F=None, phi=None, phidot=None):
        rho_ = self.get_density(rho, phi=phi)
        if phi is not None:
            rhodot_ = ufl.diff(rho_,phi) * phidot
        else:
            rhodot_ = ufl.as_ufl(0)
        return ufl.dot(rho_*a + rhodot_*v, self.var_v) * ddomain

    # Internal virtual power \delta \mathcal{P}_{\mathrm{int}}
    def deltaW_int(self, sig, ddomain, F=None):
        """ TeX:
        \int\limits_{\Omega}\boldsymbol{\sigma} : \nabla\delta\boldsymbol{v}\,\mathrm{d}v
        """
        return ufl.inner(sig, ufl.grad(self.var_v)) * ddomain

    # conservation of mass
    def deltaW_int_pres(self, v, rho, var_p, ddomain, w=None, F=None, phi=None, phidot=None):
        rho_ = self.get_density(rho, phi=phi)
        if phi is not None:
            rhodot_ = ufl.diff(rho_,phi) * phidot
        else:
            rhodot_ = ufl.as_ufl(0)
        if self.formulation == "nonconservative":
            """ TeX:
            \int\limits_{\Omega}\left(\frac{\partial\rho}{\partial t} + \nabla\rho\cdot\boldsymbol{v} + \rho\nabla\cdot\boldsymbol{v}\right)\delta p\,\mathrm{d}v
            """
            return (rhodot_ + ufl.dot(ufl.grad(rho_), v) + rho_*ufl.div(v)) * var_p * ddomain
        elif self.formulation == "conservative":
            """ TeX:
            \int\limits_{\Omega}\left(\frac{\partial\rho}{\partial t} + \nabla\cdot(\rho\boldsymbol{v})\right)\delta p \,\mathrm{d}v = 0
            """
            return (rhodot_ + ufl.div(rho_*v)) * var_p * ddomain
        else:
            raise ValueError("Unknown fluid formulation!")

    def res_v_strong_navierstokes_transient(self, a, v, rho, sig, w=None, F=None, phi=None, phidot=None):
        return self.f_inert_strong_navierstokes_transient(a, v, rho, w=w, F=F, phi=phi, phidot=phidot) - self.f_stress_strong(sig, F=F)

    def res_v_strong_navierstokes_steady(self, v, rho, sig, w=None, F=None, phi=None, phidot=None):
        return self.f_inert_strong_navierstokes_steady(v, rho, w=w, F=F, phi=phi, phidot=phidot) - self.f_stress_strong(sig, F=F)

    def res_v_strong_stokes_transient(self, a, v, rho, sig, w=None, F=None, phi=None):
        return self.f_inert_strong_stokes_transient(a, rho, phi=phi) - self.f_stress_strong(sig, F=F)

    def res_v_strong_stokes_steady(self, rho, sig, F=None):
        return -self.f_stress_strong(sig, F=F)

    def f_inert_strong_navierstokes_transient(self, a, v, rho, w=None, F=None, phi=None, phidot=None):
        rho_ = self.get_density(rho, phi=phi)
        if self.formulation == "nonconservative":
            assert(phi is None)
            return rho_ * (a + ufl.grad(v) * v)
        elif self.formulation == "conservative":
            if phi is not None:
                rhodot_ = ufl.diff(rho_,phi) * phidot
            else:
                rhodot_ = ufl.as_ufl(0)
            return rho_*a + rhodot_*v + ufl.div(rho_*ufl.outer(v, v))
        else:
            raise ValueError("Unknown fluid formulation!")

    def f_inert_strong_navierstokes_steady(self, v, rho, w=None, F=None, phi=None, phidot=None):
        rho_ = self.get_density(rho, phi=phi)
        if self.formulation == "nonconservative":
            assert(phi is None)
            return rho_ * (ufl.grad(v) * v)
        elif self.formulation == "conservative":
            return ufl.div(rho_*ufl.outer(v, v))
        else:
            raise ValueError("Unknown fluid formulation!")

    def f_inert_strong_stokes_transient(self, a, rho, phi=None, phidot=None):
        rho_ = self.get_density(rho, phi=phi)
        if phi is not None:
            rhodot_ = ufl.diff(rho_,phi) * phidot
        else:
            rhodot_ = ufl.as_ufl(0)
        return rho_*a + rhodot_*v

    def f_stress_strong(self, sig, F=None):
        return ufl.div(sig)

    def f_gradp_strong(self, p, F=None):
        return ufl.grad(p)

    def res_p_strong(self, v, rho, w=None, F=None, phi=None, phidot=None):
        rho_ = self.get_density(rho, phi=phi)
        if phi is not None:
            rhodot_ = ufl.diff(rho_,phi) * phidot
        else:
            rhodot_ = ufl.as_ufl(0)
        if self.formulation == "nonconservative":
            return rhodot_ + ufl.dot(ufl.grad(rho_), v) + rho_*ufl.div(v)
        elif self.formulation == "conservative":
            return rhodot_ + ufl.div(rho_*v)
        else:
            raise ValueError("Unknown fluid formulation!")

    # stabilized Neumann BC - Esmaily Moghadam et al. 2011
    def deltaW_ext_stabilized_neumann(self, v, beta, dboundary, w=None, F=None):
        vn = ufl.dot(v, self.n0)
        return beta * (ufl.min_value(vn, 0.0) * ufl.dot(v, self.var_v) * dboundary)

    # mod. stabilized Neumann BC
    def deltaW_ext_stabilized_neumann_mod(self, v, beta, gamma, dboundary, w=None, F=None):
        vn = ufl.dot(v, self.n0)
        return beta * (
            (vn**2.0) / (vn**2.0 + 0.01 * gamma**2.0) * ufl.min_value(vn, 0.0) * ufl.dot(v, self.var_v) * dboundary
        )  # version from Esmaily Moghadam et al. 2011 if gamma = 0

    # Robin condition for valve, over internal surface
    def deltaW_ext_robin_valve(self, v, beta, dboundary, fcts="+", w=None, F=None):
        return (-(beta * ufl.dot(v, self.var_v)))(fcts) * dboundary

    def deltaW_ext_robin_valve_deriv_visc(self, v, dboundary, fcts="+", w=None, F=None):
        return (-(ufl.dot(v, self.var_v)))(fcts) * dboundary

    # Robin condition for valve, over internal surface - normal direction
    def deltaW_ext_robin_valve_normal_ref(self, v, beta, dboundary, fcts="+", w=None, F=None):
        return (-(beta * ufl.dot(ufl.outer(self.n0, self.n0) * v, self.var_v)))(fcts) * dboundary

    ### SUPG/PSPG stabilization - cf. Tezduyar and Osawa (2000), "Finite element stabilization parameters computed from element matrices and vectors"
    def stab_supg(
        self,
        v,
        res_v_strong,
        tau_supg,
        ddomain,
        w=None,
        F=None,
        symmetric=False,
    ):
        if symmetric:  # modification to make the effective stress symmetric - experimental, use with care...
            return ufl.dot(tau_supg * ufl.sym(ufl.grad(self.var_v)) * v, res_v_strong) * ddomain
        else:
            return ufl.dot(tau_supg * ufl.grad(self.var_v) * v, res_v_strong) * ddomain

    def stab_pspg(self, var_p, res_v_strong, tau_pspg, ddomain, F=None):
        return ufl.dot(tau_pspg * ufl.grad(var_p), res_v_strong) * ddomain

    def stab_lsic(self, v, tau_lsic, rho, ddomain, w=None, F=None, phi=None, phidot=None):
        return tau_lsic * ufl.div(self.var_v) * self.res_p_strong(v, rho, w=w, F=F, phi=phi, phidot=phidot) * ddomain

    # components of element-level Reynolds number - cf. Tezduyar and Osawa (2000) - not used so far... need to assemble a cell-based vector in order to evaluate these!
    def re_c(self, rho, v, ddomain, w=None, F=None, phi=None, phidot=None):
        rho_ = self.get_density(rho, phi=phi)
        return rho_ * ufl.dot(ufl.grad(v) * v, self.var_v) * ddomain

    def re_ktilde(self, rho, v, ddomain, w=None, F=None, phi=None, phidot=None):
        rho_ = self.get_density(rho, phi=phi)
        return rho_ * ufl.dot(ufl.grad(v) * v, ufl.grad(self.var_v) * v) * ddomain

    ### Flux coupling conditions

    # flux
    # TeX: \int\limits_{\Gamma} \boldsymbol{n}\cdot\boldsymbol{v}\,\mathrm{d}a
    def flux(self, v, dboundary, w=None, F=None, fcts=None):
        if fcts is None:
            return ufl.dot(self.n0, v) * dboundary
        else:
            return (ufl.dot(self.n0, v))(fcts) * dboundary

    # get the density expression - constant or multi-phase-like
    def get_density(self, rho, phi=None):
        if phi is not None:
            return phi * rho[0] + (1.0 - phi) * rho[1]
        else:
            return rho[0]


# ALE fluid mechanics variational forms class
# Principle of Virtual Power
# TeX: \delta \mathcal{P} = \delta \mathcal{P}_{\mathrm{kin}} + \delta \mathcal{P}_{\mathrm{int}} - \delta \mathcal{P}_{\mathrm{ext}} = 0, \quad \forall \; \delta\boldsymbol{v}

# gradients of a vector field transform according to:
# grad(u) = Grad(u) * F^(-1)
# gradients of a scalar field transform according to:
# grad(p) = F^(-T) * Grad(p)
# divergences of a tensor/vector field transform according to:
# div(A) = Grad(A) : F^(-T)

# Piola identity for a vector field:
# J div(a) = Div(J F^{-1} a)
# Piola identity for a second-order tensor field:
# J div(A) = Div(J A F^{-T})

class variationalform_ale(variationalform):
    # Kinetic virtual power \delta \mathcal{P}_{\mathrm{kin}}
    def deltaW_kin_navierstokes_transient(self, a, v, rho, ddomain, w=None, F=None, phi=None, phidot=None):
        J = ufl.det(F)
        rho_ = self.get_density(rho, phi=phi)
        if self.formulation == "nonconservative":
            assert(phi is None)
            """ TeX:
            \int\limits_{\Omega_0}\widehat{J}\rho\left(\left.\frac{\partial\boldsymbol{v}}{\partial t}\right|_{\boldsymbol{x}_0} + (\nabla_0\boldsymbol{v}\widehat{\boldsymbol{F}}^{-1})(\boldsymbol{v}-\boldsymbol{w})\right)\cdot\delta \boldsymbol{v}\,\mathrm{d}V
            """
            return J*rho_ * ufl.dot(a + ufl.grad(v) * ufl.inv(F) * (v - w), self.var_v) * ddomain
        elif self.formulation == "conservative":
            Jdot = ufl.div(J*ufl.inv(F)*w)
            if phi is not None:
                rhodot_ = ufl.diff(rho_,phi) * phidot
            else:
                rhodot_ = ufl.as_ufl(0)
            """ TeX:
            \int\limits_{\Omega_0}\left(\left.\frac{\partial(\widehat{J}\rho\boldsymbol{v})}{\partial t}\right|_{\boldsymbol{x}_0} + \nabla_0\cdot\left(\widehat{J}\rho((\boldsymbol{v}-\boldsymbol{w})\otimes\boldsymbol{v})\widehat{\boldsymbol{F}}^{-\mathrm{T}}\right)\right)\cdot\delta \boldsymbol{v}\,\mathrm{d}V
            """
            return ufl.dot((J*rhodot_ + rho_*Jdot)*v + J*rho_*a + ufl.div(J*rho_*ufl.outer(v - w, v)*ufl.inv(F).T), self.var_v) * ddomain
        else:
            raise ValueError("Unknown fluid formulation! Choose either 'nonconservative' or 'conservative'.")

    def deltaW_kin_navierstokes_steady(self, v, rho, ddomain, w=None, F=None, phi=None, phidot=None):
        J = ufl.det(F)
        rho_ = self.get_density(rho, phi=phi)
        if self.formulation == "nonconservative":
            assert(phi is None)
            """ TeX:
            \int\limits_{\Omega_0}\widehat{J}\rho(\nabla_0\boldsymbol{v}\boldsymbol{F}^{-1})\boldsymbol{v}\cdot\delta \boldsymbol{v}\,\mathrm{d}V
            """
            return J*rho_ * ufl.dot(ufl.grad(v) * ufl.inv(F) * v, self.var_v) * ddomain  # NOTE: No domain velocity here! ... Really?!
        elif self.formulation == "conservative":
            """ TeX:
            \int\limits_{\Omega_0}\nabla_0\cdot\left(\widehat{J}\rho(\boldsymbol{v}\otimes\boldsymbol{v})\widehat{\boldsymbol{F}}^{-\mathrm{T}}\right)\cdot\delta \boldsymbol{v}\,\mathrm{d}V
            """
            return ufl.dot(ufl.div(J*rho_*ufl.outer(v, v)*ufl.inv(F).T), self.var_v) * ddomain  # NOTE: No domain velocity here! ... Really?!
        else:
            raise ValueError("Unknown fluid formulation!")

    def deltaW_kin_stokes_transient(self, a, v, rho, ddomain, w=None, F=None, phi=None, phidot=None):
        J = ufl.det(F)
        rho_ = self.get_density(rho, phi=phi)
        if self.formulation == "nonconservative":
            assert(phi is None)
            """ TeX:
            \int\limits_{\Omega_0}\widehat{J}\rho\left(\left.\frac{\partial\boldsymbol{v}}{\partial t}\right|_{\boldsymbol{x}_0} + (\nabla_0\boldsymbol{v}\widehat{\boldsymbol{F}}^{-1})(-\boldsymbol{w})\right)\cdot\delta \boldsymbol{v}\,\mathrm{d}V
            """
            return J*rho_ * ufl.dot(a + ufl.grad(v) * ufl.inv(F) * (-w), self.var_v) * ddomain
        elif self.formulation == "conservative":
            Jdot = ufl.div(J*ufl.inv(F)*w)
            if phi is not None:
                rhodot_ = ufl.diff(rho_,phi) * phidot
            else:
                rhodot_ = ufl.as_ufl(0)
            """ TeX:
            \int\limits_{\Omega_0}\left(\left.\frac{\partial(\widehat{J}\rho\boldsymbol{v})}{\partial t}\right|_{\boldsymbol{x}_0} + \nabla_0\cdot\left(\widehat{J}\rho((-\boldsymbol{w})\otimes\boldsymbol{v})\widehat{\boldsymbol{F}}^{-\mathrm{T}}\right)\right)\cdot\delta \boldsymbol{v}\,\mathrm{d}V
            """
            return ufl.dot((J*rhodot_ + rho_*Jdot)*v + J*rho_*a + ufl.div(J*rho_*ufl.outer(-w, v)*ufl.inv(F).T), self.var_v) * ddomain
        else:
            raise ValueError("Unknown fluid formulation!")

    # Internal virtual power \delta \mathcal{P}_{\mathrm{int}}
    def deltaW_int(self, sig, ddomain, F=None):
        J = ufl.det(F)
        """ TeX:
        \int\limits_{\Omega_0}\widehat{J}\boldsymbol{\sigma}\widehat{\boldsymbol{F}}^{-\mathrm{T}} : \nabla_0\delta\boldsymbol{v}\,\mathrm{d}V
        """
        return ufl.inner(J*sig*ufl.inv(F).T, ufl.grad(self.var_v)) * ddomain

    # conservation of mass in ALE form
    def deltaW_int_pres(self, v, rho, var_p, ddomain, w=None, F=None, phi=None, phidot=None):
        J = ufl.det(F)
        rho_ = self.get_density(rho, phi=phi)
        Jdot = ufl.div(J*ufl.inv(F)*w)
        if phi is not None:
            rhodot_ = ufl.diff(rho_,phi) * phidot
        else:
            rhodot_ = ufl.as_ufl(0)
        if self.formulation == "nonconservative":
            """ TeX:
            \int\limits_{\Omega_0}\left(\left.\frac{\partial(\widehat{J}\rho)}{\partial t}\right|_{\boldsymbol{x}_0} + \widehat{J}\widehat{\boldsymbol{F}}^{-1}\nabla_0\rho\cdot(\boldsymbol{v}-\boldsymbol{w}) + \rho\nabla_0\cdot\left(\widehat{J}\widehat{\boldsymbol{F}}^{-1}(\boldsymbol{v}-\boldsymbol{w})\right)\right)\delta p\,\mathrm{d}V
            """
            return (J*rhodot_ + rho_*Jdot + J*ufl.dot(ufl.inv(F).T*ufl.grad(rho_), v-w) + rho_*ufl.div(J*ufl.inv(F)*(v-w))) * var_p * ddomain
        elif self.formulation == "conservative":
            """ TeX:
            \int\limits_{\Omega_0}\left(\left.\frac{\partial(\widehat{J}\rho)}{\partial t}\right|_{\boldsymbol{x}_0} + \nabla_0\cdot\left(\widehat{J}\widehat{\boldsymbol{F}}^{-1}\rho(\boldsymbol{v}-\boldsymbol{w})\right)\right)\delta p\,\mathrm{d}V
            """
            return (J*rhodot_ + rho_*Jdot + ufl.div(J*ufl.inv(F)*rho_*(v-w))) * var_p * ddomain
        else:
            raise ValueError("Unknown fluid formulation!")

    def res_v_strong_navierstokes_transient(self, a, v, rho, sig, w=None, F=None, phi=None, phidot=None):
        return self.f_inert_strong_navierstokes_transient(a, v, rho, w=w, F=F, phi=phi, phidot=phidot) - self.f_stress_strong(sig, F=F)

    def res_v_strong_navierstokes_steady(self, v, rho, sig, w=None, F=None, phi=None, phidot=None):
        return self.f_inert_strong_navierstokes_steady(v, rho, w=w, F=F, phi=phi, phidot=phidot) - self.f_stress_strong(sig, F=F)

    def res_v_strong_stokes_transient(self, a, v, rho, sig, w=None, F=None, phi=None, phidot=None):
        return self.f_inert_strong_stokes_transient(a, v, rho, w=w, F=F, phi=phi, phidot=phidot) - self.f_stress_strong(sig, F=F)

    def res_v_strong_stokes_steady(self, rho, sig, F=None):
        return -self.f_stress_strong(sig, F=F)

    def f_inert_strong_navierstokes_transient(self, a, v, rho, w=None, F=None, phi=None, phidot=None):
        J = ufl.det(F)
        rho_ = self.get_density(rho, phi=phi)
        if self.formulation == "nonconservative":
            assert(phi is None)
            return J*rho_ * (a + ufl.grad(v) * ufl.inv(F) * (v - w))
        elif self.formulation == "conservative":
            Jdot = ufl.div(J*ufl.inv(F)*w)
            if phi is not None:
                rhodot_ = ufl.diff(rho_,phi) * phidot
            else:
                rhodot_ = ufl.as_ufl(0)
            return (J*rhodot_ + rho_*Jdot)*v + J*rho_*a + ufl.div(J*rho_*ufl.outer(v - w, v)*ufl.inv(F).T)
        else:
            raise ValueError("Unknown fluid formulation!")

    def f_inert_strong_navierstokes_steady(self, v, rho, w=None, F=None, phi=None, phidot=None):
        J = ufl.det(F)
        rho_ = self.get_density(rho, phi=phi)
        if self.formulation == "nonconservative":
            assert(phi is None)
            return J*rho_ * (ufl.grad(v) * ufl.inv(F) * v)  # NOTE: No domain velocity here! ... Really?!
        elif self.formulation == "conservative":
            return ufl.div(J*rho_*ufl.outer(v, v)*ufl.inv(F).T)  # NOTE: No domain velocity here! ... Really?!
        else:
            raise ValueError("Unknown fluid formulation!")

    def f_inert_strong_stokes_transient(self, a, v, rho, w=None, F=None, phi=None, phidot=None):
        J = ufl.det(F)
        rho_ = self.get_density(rho, phi=phi)
        if self.formulation == "nonconservative":
            assert(phi is None)
            return J*rho_ * (a + ufl.grad(v) * ufl.inv(F) * (-w))
        elif self.formulation == "conservative":
            Jdot = ufl.div(J*ufl.inv(F)*w)
            if phi is not None:
                rhodot_ = ufl.diff(rho_,phi) * phidot
            else:
                rhodot_ = ufl.as_ufl(0)
            return (J*rhodot_ + rho_*Jdot)*v + J*rho_*a + ufl.div(J*rho_*ufl.outer(-w, v)*ufl.inv(F).T)
        else:
            raise ValueError("Unknown fluid formulation!")

    def f_stress_strong(self, sig, F=None):
        J = ufl.det(F)
        return ufl.div(J*sig*ufl.inv(F).T)

    def f_gradp_strong(self, p, F=None):
        J = ufl.det(F)
        return J*ufl.inv(F).T * ufl.grad(p)

    def res_p_strong(self, v, rho, w=None, F=None, phi=None, phidot=None):
        J = ufl.det(F)
        rho_ = self.get_density(rho, phi=phi)
        Jdot = ufl.div(J*ufl.inv(F)*w)
        if phi is not None:
            rhodot_ = ufl.diff(rho_,phi) * phidot
        else:
            rhodot_ = ufl.as_ufl(0)
        if self.formulation == "nonconservative":
            return J*rhodot_ + rho_*Jdot + J*ufl.dot(ufl.inv(F).T*ufl.grad(rho_), v-w) + rho_*ufl.div(J*ufl.inv(F)*(v-w))
        elif self.formulation == "conservative":
            return J*rhodot_ + rho_*Jdot + ufl.div(J*ufl.inv(F)*rho_*(v-w))
        else:
            raise ValueError("Unknown fluid formulation!")

    # stabilized Neumann BC - Esmaily Moghadam et al. 2011
    def deltaW_ext_stabilized_neumann(self, v, beta, dboundary, w=None, F=None):
        J = ufl.det(F)
        vwn = ufl.dot(v - w, J * ufl.inv(F).T * self.n0)
        return beta * (ufl.min_value(vwn, 0.0) * ufl.dot(v, self.var_v) * dboundary)

    # mod. stabilized Neumann BC
    def deltaW_ext_stabilized_neumann_mod(self, v, beta, gamma, dboundary, w=None, F=None):
        J = ufl.det(F)
        vwn = ufl.dot(v - w, J * ufl.inv(F).T * self.n0)
        return beta * (
            (vwn**2.0) / (vwn**2.0 + 0.01 * gamma**2.0) * ufl.min_value(vwn, 0.0) * ufl.dot(v, self.var_v) * dboundary
        )  # version from Esmaily Moghadam et al. 2011 if gamma = 0

    # Robin condition for valve, over internal surface
    def deltaW_ext_robin_valve(self, v, beta, dboundary, fcts="+", w=None, F=None):
        return (-(beta * ufl.dot((v - w), self.var_v)))(fcts) * dboundary

    def deltaW_ext_robin_valve_deriv_visc(self, v, dboundary, fcts="+", w=None, F=None):
        return (-(ufl.dot((v - w), self.var_v)))(fcts) * dboundary

    # Robin condition for valve, over internal surface - normal direction
    def deltaW_ext_robin_valve_normal_ref(self, v, beta, dboundary, fcts="+", w=None, F=None):
        return (-(beta * ufl.dot(ufl.outer(self.n0, self.n0) * (v - w), self.var_v)))(fcts) * dboundary

    ### SUPG/PSPG stabilization
    def stab_supg(
        self,
        v,
        res_v_strong,
        tau_supg,
        ddomain,
        w=None,
        F=None,
        symmetric=False,
    ):
        # NOTE: J=det(F) already included in res_v_strong
        if symmetric:  # modification to make the effective stress symmetric - experimental, use with care...
            return ufl.dot(tau_supg * ufl.sym(ufl.grad(self.var_v) * ufl.inv(F)) * v, res_v_strong) * ddomain
        else:
            return ufl.dot(tau_supg * ufl.grad(self.var_v) * ufl.inv(F) * v, res_v_strong) * ddomain

    def stab_pspg(self, var_p, res_v_strong, tau_pspg, ddomain, F=None):
        # NOTE: J=det(F) already included in res_v_strong
        return ufl.dot(tau_pspg * ufl.inv(F).T * ufl.grad(var_p), res_v_strong) * ddomain

    def stab_lsic(self, v, tau_lsic, rho, ddomain, w=None, F=None, phi=None, phidot=None):
        # NOTE: J=det(F) already included in res_p_strong
        return tau_lsic * ufl.inner(ufl.grad(self.var_v), ufl.inv(F).T) * self.res_p_strong(v, rho, w=w, F=F, phi=phi, phidot=phidot) * ddomain

    # components of element-level Reynolds number - not used so far... need to assemble a cell-based vector in order to evaluate these!
    def re_c(self, rho, v, ddomain, w=None, F=None, phi=None, phidot=None):
        rho_ = self.get_density(rho, phi=phi)
        J = ufl.det(F)
        return rho_ * ufl.dot(ufl.grad(v) * ufl.inv(F) * (v - w), self.var_v) * J * ddomain

    def re_ktilde(self, rho, v, ddomain, w=None, F=None, phi=None, phidot=None):
        rho_ = self.get_density(rho, phi=phi)
        J = ufl.det(F)
        return (
            rho_
            * ufl.dot(
                ufl.grad(v) * ufl.inv(F) * (v - w),
                ufl.grad(self.var_v) * ufl.inv(F) * v,
            )
            * J
            * ddomain
        )

    ### Flux coupling conditions

    # flux
    # TeX: \int\limits_{\Gamma} (\boldsymbol{v}-\boldsymbol{w})\cdot\boldsymbol{n}\,\mathrm{d}a =
    #      \int\limits_{\Gamma_0} (\boldsymbol{v}-\boldsymbol{w})\cdot J\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_0\,\mathrm{d}A
    def flux(self, v, dboundary, w=None, F=None, fcts=None):
        J = ufl.det(F)
        if fcts is None:
            return J * ufl.dot(ufl.inv(F).T * self.n0, (v - w)) * dboundary
        else:
            return (J * ufl.dot(ufl.inv(F).T * self.n0, (v - w)))(fcts) * dboundary
