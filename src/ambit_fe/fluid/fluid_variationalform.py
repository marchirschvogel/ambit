#!/usr/bin/env python3

# Copyright (c) 2019-2026, Dr.-Ing. Marc Hirschvogel
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
    def __init__(
        self,
        tstfnc1=None,
        tstfnc2=None,
        trlfnc1=None,
        trlfnc2=None,
        n0=None,
        x_ref=None,
        formulation=None,
        ro0=None,
    ):
        self.var_v = tstfnc1
        self.var_p = tstfnc2
        variationalform_base.__init__(self, tstfnc1=tstfnc1, tstfnc2=tstfnc2, trlfnc1=trlfnc1, trlfnc2=trlfnc2, n0=n0, x_ref=x_ref, ro0=ro0)
        self.formulation, self.mass_formulation = formulation[0], formulation[1]
        self.I = ufl.Identity(len(self.var_v))

    # Kinetic virtual power \delta \mathcal{P}_{\mathrm{kin}}
    def deltaW_kin_navierstokes_transient(self, a, v, rho, ddomain, w=None, F=None, phi=[None,None], phidot=None):
        rho_ = self.get_density(rho, chi=phi[1])
        # standard Eulerian fluid
        if self.formulation == "nonconservative":
            assert(phi[0] is None)
            """ TeX:
            \int\limits_{\mathit{\Omega}} \rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + (\nabla\boldsymbol{v})\boldsymbol{v}\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}V
            """
            return rho_ * ufl.dot(a + ufl.grad(v) * v, self.var_v) * ddomain
        elif self.formulation == "conservative":
            if phi[0] is not None:
                rhodot_ = ufl.diff(rho_,phi[0]) * phidot
            else:
                rhodot_ = ufl.as_ufl(0)
            """ TeX:
            \int\limits_{\mathit{\Omega}} \left(\frac{\partial(\rho\boldsymbol{v})}{\partial t} + \nabla\cdot(\rho(\boldsymbol{v}\otimes\boldsymbol{v}))\right) \cdot \delta\boldsymbol{v} \,\mathrm{d}V
            """
            return ufl.dot(rho_*a + rhodot_*v + ufl.div(rho_*ufl.outer(v, v)), self.var_v) * ddomain
        else:
            raise ValueError("Unknown fluid formulation! Choose either 'nonconservative' or 'conservative'.")

    def deltaW_kin_navierstokes_steady(self, v, rho, ddomain, F=None, phi=[None,None], phidot=None):
        rho_ = self.get_density(rho, chi=phi[1])
        if self.formulation == "nonconservative":
            assert(phi[0] is None)
            """ TeX:
            \int\limits_{\mathit{\Omega}} \rho (\nabla\boldsymbol{v})\boldsymbol{v} \cdot \delta\boldsymbol{v} \,\mathrm{d}V
            """
            return rho_ * ufl.dot(ufl.grad(v) * v, self.var_v) * ddomain
        elif self.formulation == "conservative":
            """ TeX:
            \int\limits_{\mathit{\Omega}} \nabla\cdot(\rho(\boldsymbol{v}\otimes\boldsymbol{v})) \cdot \delta\boldsymbol{v} \,\mathrm{d}V
            """
            return ufl.dot(ufl.div(rho_*ufl.outer(v, v)), self.var_v) * ddomain
        else:
            raise ValueError("Unknown fluid formulation!")

    def deltaW_kin_stokes_transient(self, a, v, rho, ddomain, w=None, F=None, phi=[None,None], phidot=None):
        rho_ = self.get_density(rho, chi=phi[1])
        if phi[0] is not None:
            rhodot_ = ufl.diff(rho_,phi[0]) * phidot
        else:
            rhodot_ = ufl.as_ufl(0)
        return ufl.dot(rho_*a + rhodot_*v, self.var_v) * ddomain

    # Internal virtual power \delta \mathcal{P}_{\mathrm{int}}
    def deltaW_int(self, sig, ddomain, F=None):
        """ TeX:
        \int\limits_{\mathit{\Omega}}\boldsymbol{\sigma} : \nabla\delta\boldsymbol{v}\,\mathrm{d}V
        """
        return ufl.inner(sig, ufl.grad(self.var_v)) * ddomain

    # conservation of mass
    def deltaW_int_pres(self, v, var_p, ddomain, rho=None, w=None, F=None, phi=[None,None], phidot=None):
        if self.mass_formulation=="conservative_mass":
            rho_ = self.get_density(rho, chi=phi[1])
            if phi[0] is not None:
                rhodot_ = ufl.diff(rho_,phi[0]) * phidot
            else:
                rhodot_ = ufl.as_ufl(0)
            if self.formulation == "nonconservative":
                """ TeX:
                \int\limits_{\mathit{\Omega}}\left(\frac{\partial\rho}{\partial t} + \nabla\rho\cdot\boldsymbol{v} + \rho\nabla\cdot\boldsymbol{v}\right)\delta p\,\mathrm{d}V = 0
                """
                return (rhodot_ + ufl.dot(ufl.grad(rho_), v) + rho_*ufl.div(v)) * var_p * ddomain
            elif self.formulation == "conservative":
                """ TeX:
                \int\limits_{\mathit{\Omega}}\left(\frac{\partial\rho}{\partial t} + \nabla\cdot(\rho\boldsymbol{v})\right)\delta p \,\mathrm{d}V = 0
                """
                return (rhodot_ + ufl.div(rho_*v)) * var_p * ddomain
            else:
                raise ValueError("Unknown fluid formulation!")
        elif self.mass_formulation=="reduced_mass":
            """ TeX:
            \int\limits_{\mathit{\Omega}}(\nabla\cdot\boldsymbol{v}\,\delta p)\,\mathrm{d}V = 0
            """
            return ufl.div(v) * var_p * ddomain
        else:
            raise ValueError("Unknown fluid mass formulation!")

    # CH-NS part for reduced mass formulation
    def deltaW_int_pres_reduced_ch(self, alpha, Jflux, var_p, ddomain, F=None):
        """ TeX:
        \int\limits_{\mathit{\Omega}}(\alpha\,\boldsymbol{J}\cdot\nabla\delta p)\,\mathrm{d}V = 0
        """
        return ufl.inner(alpha*Jflux, ufl.grad(var_p)) * ddomain

    def res_v_strong_navierstokes_transient(self, a, v, rho, sig, fbody, w=None, F=None, phi=[None,None], phidot=None):
        return self.f_inert_strong_navierstokes_transient(a, v, rho, w=w, F=F, phi=phi, phidot=phidot) - self.f_stress_strong(sig, F=F) - fbody

    def res_v_strong_navierstokes_steady(self, v, rho, sig, fbody, w=None, F=None, phi=[None,None], phidot=None):
        return self.f_inert_strong_navierstokes_steady(v, rho, w=w, F=F, phi=phi, phidot=phidot) - self.f_stress_strong(sig, F=F) - fbody

    def res_v_strong_stokes_transient(self, a, v, rho, sig, fbody, w=None, F=None, phi=[None,None]):
        return self.f_inert_strong_stokes_transient(a, rho, phi=phi) - self.f_stress_strong(sig, F=F) - fbody

    def res_v_strong_stokes_steady(self, rho, sig, fbody, F=None):
        return -self.f_stress_strong(sig, F=F) - fbody

    def f_inert_strong_navierstokes_transient(self, a, v, rho, w=None, F=None, phi=[None,None], phidot=None):
        rho_ = self.get_density(rho, chi=phi[1])
        if self.formulation == "nonconservative":
            assert(phi[0] is None)
            return rho_ * (a + ufl.grad(v) * v)
        elif self.formulation == "conservative":
            if phi[0] is not None:
                rhodot_ = ufl.diff(rho_,phi[0]) * phidot
            else:
                rhodot_ = ufl.as_ufl(0)
            return rho_*a + rhodot_*v + ufl.div(rho_*ufl.outer(v, v))
        else:
            raise ValueError("Unknown fluid formulation!")

    def f_inert_strong_navierstokes_steady(self, v, rho, w=None, F=None, phi=[None,None], phidot=None):
        rho_ = self.get_density(rho, chi=phi[1])
        if self.formulation == "nonconservative":
            assert(phi[0] is None)
            return rho_ * (ufl.grad(v) * v)
        elif self.formulation == "conservative":
            return ufl.div(rho_*ufl.outer(v, v))
        else:
            raise ValueError("Unknown fluid formulation!")

    def f_inert_strong_stokes_transient(self, a, rho, phi=[None,None], phidot=None):
        rho_ = self.get_density(rho, chi=phi[1])
        if phi[0] is not None:
            rhodot_ = ufl.diff(rho_,phi[0]) * phidot
        else:
            rhodot_ = ufl.as_ufl(0)
        return rho_*a + rhodot_*v

    def f_stress_strong(self, sig, F=None):
        return ufl.div(sig)

    def f_gradp_strong(self, p, F=None):
        return ufl.grad(p)

    # NOTE: For grad/div stabilization, we only use the "reduced mass" residual version here! However, needs investigation...!
    def res_p_strong(self, v, rho, w=None, F=None, phi=[None,None], phidot=None):
        if self.mass_formulation=="conservative_mass":
            rho_ = self.get_density(rho, chi=phi[1])
            if phi[0] is not None:
                rhodot_ = ufl.diff(rho_,phi[0]) * phidot
            else:
                rhodot_ = ufl.as_ufl(0)
            if self.formulation == "nonconservative":
                # return rhodot_ + ufl.dot(ufl.grad(rho_), v) + rho_*ufl.div(v)
                return rho_*ufl.div(v)
            elif self.formulation == "conservative":
                # return rhodot_ + ufl.div(rho_*v)
                return rho_*ufl.div(v)
            else:
                raise ValueError("Unknown fluid formulation!")
        elif self.mass_formulation=="reduced_mass":
            return ufl.div(v)
        else:
            raise ValueError("Unknown fluid mass formulation!")

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

    def stab_pspg(self, var_p, res_v_strong, tau_pspg, rho, ddomain, F=None, phi=[None,None]):
        if self.mass_formulation=="conservative_mass":
            return ufl.dot(tau_pspg * ufl.grad(var_p), res_v_strong) * ddomain
        elif self.mass_formulation=="reduced_mass":
            rho_ = self.get_density(rho, chi=phi[1])
            return (1./rho_)*ufl.dot(tau_pspg * ufl.grad(var_p), res_v_strong) * ddomain
        else:
            raise ValueError("Unknown fluid mass formulation!")

    def stab_lsic(self, v, tau_lsic, rho, ddomain, w=None, F=None, phi=[None,None], phidot=None):
        if self.mass_formulation=="conservative_mass":
            return tau_lsic * ufl.div(self.var_v) * self.res_p_strong(v, rho, w=w, F=F, phi=phi, phidot=phidot) * ddomain
        elif self.mass_formulation=="reduced_mass":
            # rho_ = self.get_density(rho, chi=phi[1])
            rho_ = sum(rho) / len(rho)  # let's use the average here so far...
            return tau_lsic * ufl.div(self.var_v) * rho_ * self.res_p_strong(v, rho, w=w, F=F, phi=phi, phidot=phidot) * ddomain
        else:
            raise ValueError("Unknown fluid mass formulation!")

    # components of element-level Reynolds number - cf. Tezduyar and Osawa (2000) - not used so far... need to assemble a cell-based vector in order to evaluate these!
    def re_c(self, rho, v, ddomain, w=None, F=None, phi=[None,None], phidot=None):
        rho_ = self.get_density(rho, chi=phi[1])
        return rho_ * ufl.dot(ufl.grad(v) * v, self.var_v) * ddomain

    def re_ktilde(self, rho, v, ddomain, w=None, F=None, phi=[None,None], phidot=None):
        rho_ = self.get_density(rho, chi=phi[1])
        return rho_ * ufl.dot(ufl.grad(v) * v, ufl.grad(self.var_v) * v) * ddomain

    ### Flux coupling conditions

    # flux
    # TeX: \int\limits_{\mathit{\Gamma}} \boldsymbol{n}\cdot\boldsymbol{v}\,\mathrm{d}A
    def flux(self, v, dboundary, w=None, F=None, fcts=None):
        if fcts is None:
            return ufl.dot(self.n0, v) * dboundary
        else:
            return (ufl.dot(self.n0, v))(fcts) * dboundary

    # capillary force in multiphase flow
    def capillary_force(self, phi, mu, ddomain, F=None, return_type="work"):
        """ TeX:
        \int\limits_{\mathit{\Omega}}\phi\nabla\mu \cdot \delta\boldsymbol{v} \,\mathrm{d}V
        """
        return ufl.dot(phi * ufl.grad(mu), self.var_v) * ddomain

    # Generalized form of Korteweg stress in multiphase flow
    def korteweg_stress(self, phi, mu, psi, kappa, ddomain, F=None, return_type="work"):
        """ TeX:
        \int\limits_{\mathit{\Omega}} \left(\kappa\nabla\phi\otimes\nabla\phi + \left(\mu\phi - \psi -\frac{1}{2}\kappa\nabla\phi\cdot\nabla\phi\right)\boldsymbol{I}\right) : \nabla\delta\boldsymbol{v}\,\mathrm{d}V
        """
        return ufl.inner( (kappa*ufl.outer(ufl.grad(phi),ufl.grad(phi)) + (mu*phi - psi - 0.5*kappa*ufl.dot(ufl.grad(phi),ufl.grad(phi)))*self.I), ufl.grad(self.var_v)) * ddomain

    # alpha for consistent mass-averaged CH-NS
    def get_alpha_chns(self, rho, phi):
        drho = ufl.diff(rho, phi)
        return -drho / (rho - drho*phi)

    def get_alpha_chns_(self, rho):
        return -(rho[1]-rho[0])/(rho[0]+rho[1])

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
    def deltaW_kin_navierstokes_transient(self, a, v, rho, ddomain, w=None, F=None, phi=[None,None], phidot=None):
        J = ufl.det(F)
        rho_ = self.get_density(rho, chi=phi[1])
        if self.formulation == "nonconservative":
            assert(phi[0] is None)
            """ TeX:
            \int\limits_{\mathit{\Omega}_0}\widehat{J}\rho\left(\left.\frac{\partial\boldsymbol{v}}{\partial t}\right|_{\boldsymbol{x}_0} + (\nabla_0\boldsymbol{v}\widehat{\boldsymbol{F}}^{-1})(\boldsymbol{v}-\widehat{\boldsymbol{w}})\right)\cdot\delta \boldsymbol{v}\,\mathrm{d}V
            """
            return J*rho_ * ufl.dot(a + ufl.grad(v) * ufl.inv(F) * (v - w), self.var_v) * ddomain
        elif self.formulation == "conservative":
            Jdot = ufl.div(J*ufl.inv(F)*w)
            if phi[0] is not None:
                rhodot_ = ufl.diff(rho_,phi[0]) * phidot
            else:
                rhodot_ = ufl.as_ufl(0)
            """ TeX:
            \int\limits_{\mathit{\Omega}_0}\left(\left.\frac{\partial(\widehat{J}\rho\boldsymbol{v})}{\partial t}\right|_{\boldsymbol{x}_0} + \nabla_0\cdot\left(\widehat{J}\rho(\boldsymbol{v}\otimes(\boldsymbol{v}-\widehat{\boldsymbol{w}}))\widehat{\boldsymbol{F}}^{-\mathrm{T}}\right)\right)\cdot\delta \boldsymbol{v}\,\mathrm{d}V
            """
            return ufl.dot((J*rhodot_ + rho_*Jdot)*v + J*rho_*a + ufl.div(J*rho_*ufl.outer(v, v - w)*ufl.inv(F).T), self.var_v) * ddomain
        else:
            raise ValueError("Unknown fluid formulation! Choose either 'nonconservative' or 'conservative'.")

    def deltaW_kin_navierstokes_steady(self, v, rho, ddomain, w=None, F=None, phi=[None,None], phidot=None):
        J = ufl.det(F)
        rho_ = self.get_density(rho, chi=phi[1])
        if self.formulation == "nonconservative":
            assert(phi[0] is None)
            """ TeX:
            \int\limits_{\mathit{\Omega}_0}\widehat{J}\rho(\nabla_0\boldsymbol{v}\boldsymbol{F}^{-1})\boldsymbol{v}\cdot\delta \boldsymbol{v}\,\mathrm{d}V
            """
            return J*rho_ * ufl.dot(ufl.grad(v) * ufl.inv(F) * v, self.var_v) * ddomain  # NOTE: No domain velocity here! ... Really?!
        elif self.formulation == "conservative":
            """ TeX:
            \int\limits_{\mathit{\Omega}_0}\nabla_0\cdot\left(\widehat{J}\rho(\boldsymbol{v}\otimes\boldsymbol{v})\widehat{\boldsymbol{F}}^{-\mathrm{T}}\right)\cdot\delta \boldsymbol{v}\,\mathrm{d}V
            """
            return ufl.dot(ufl.div(J*rho_*ufl.outer(v, v)*ufl.inv(F).T), self.var_v) * ddomain  # NOTE: No domain velocity here! ... Really?!
        else:
            raise ValueError("Unknown fluid formulation!")

    def deltaW_kin_stokes_transient(self, a, v, rho, ddomain, w=None, F=None, phi=[None,None], phidot=None):
        J = ufl.det(F)
        rho_ = self.get_density(rho, chi=phi[1])
        if self.formulation == "nonconservative":
            assert(phi[0] is None)
            """ TeX:
            \int\limits_{\mathit{\Omega}_0}\widehat{J}\rho\left(\left.\frac{\partial\boldsymbol{v}}{\partial t}\right|_{\boldsymbol{x}_0} + (\nabla_0\boldsymbol{v}\widehat{\boldsymbol{F}}^{-1})(-\widehat{\boldsymbol{w}})\right)\cdot\delta \boldsymbol{v}\,\mathrm{d}V
            """
            return J*rho_ * ufl.dot(a + ufl.grad(v) * ufl.inv(F) * (-w), self.var_v) * ddomain
        elif self.formulation == "conservative":
            Jdot = ufl.div(J*ufl.inv(F)*w)
            if phi[0] is not None:
                rhodot_ = ufl.diff(rho_,phi[0]) * phidot
            else:
                rhodot_ = ufl.as_ufl(0)
            """ TeX:
            \int\limits_{\mathit{\Omega}_0}\left(\left.\frac{\partial(\widehat{J}\rho\boldsymbol{v})}{\partial t}\right|_{\boldsymbol{x}_0} + \nabla_0\cdot\left(\widehat{J}\rho(\boldsymbol{v}\otimes(-\widehat{\boldsymbol{w}}))\widehat{\boldsymbol{F}}^{-\mathrm{T}}\right)\right)\cdot\delta \boldsymbol{v}\,\mathrm{d}V
            """
            return ufl.dot((J*rhodot_ + rho_*Jdot)*v + J*rho_*a + ufl.div(J*rho_*ufl.outer(v, -w)*ufl.inv(F).T), self.var_v) * ddomain
        else:
            raise ValueError("Unknown fluid formulation!")

    # Internal virtual power \delta \mathcal{P}_{\mathrm{int}}
    def deltaW_int(self, sig, ddomain, F=None):
        J = ufl.det(F)
        """ TeX:
        \int\limits_{\mathit{\Omega}_0}\widehat{J}\boldsymbol{\sigma}\widehat{\boldsymbol{F}}^{-\mathrm{T}} : \nabla_0\delta\boldsymbol{v}\,\mathrm{d}V
        """
        return ufl.inner(J*sig*ufl.inv(F).T, ufl.grad(self.var_v)) * ddomain

    # conservation of mass in ALE form
    def deltaW_int_pres(self, v, var_p, ddomain, rho=None, w=None, F=None, phi=[None,None], phidot=None):
        J = ufl.det(F)
        if self.mass_formulation=="conservative_mass":
            rho_ = self.get_density(rho, chi=phi[1])
            Jdot = ufl.div(J*ufl.inv(F)*w)
            if phi[0] is not None:
                rhodot_ = ufl.diff(rho_,phi[0]) * phidot
            else:
                rhodot_ = ufl.as_ufl(0)
            if self.formulation == "nonconservative":
                """ TeX:
                \int\limits_{\mathit{\Omega}_0}\left(\left.\frac{\partial(\widehat{J}\rho)}{\partial t}\right|_{\boldsymbol{x}_0} + \widehat{J}\widehat{\boldsymbol{F}}^{-1}\nabla_0\rho\cdot(\boldsymbol{v}-\widehat{\boldsymbol{w}}) + \rho\nabla_0\cdot\left(\widehat{J}\widehat{\boldsymbol{F}}^{-1}(\boldsymbol{v}-\widehat{\boldsymbol{w}})\right)\right)\delta p\,\mathrm{d}V = 0
                """
                return (J*rhodot_ + rho_*Jdot + J*ufl.dot(ufl.inv(F).T*ufl.grad(rho_), v-w) + rho_*ufl.div(J*ufl.inv(F)*(v-w))) * var_p * ddomain
            elif self.formulation == "conservative":
                """ TeX:
                \int\limits_{\mathit{\Omega}_0}\left(\left.\frac{\partial(\widehat{J}\rho)}{\partial t}\right|_{\boldsymbol{x}_0} + \nabla_0\cdot\left(\widehat{J}\widehat{\boldsymbol{F}}^{-1}\rho(\boldsymbol{v}-\widehat{\boldsymbol{w}})\right)\right)\delta p\,\mathrm{d}V = 0
                """
                return (J*rhodot_ + rho_*Jdot + ufl.div(J*ufl.inv(F)*rho_*(v-w))) * var_p * ddomain
            else:
                raise ValueError("Unknown fluid formulation!")
        elif self.mass_formulation=="reduced_mass":
            """ TeX:
            \int\limits_{\mathit{\Omega}_0}\nabla_0\cdot(\widehat{J}\boldsymbol{F}^{-1}\boldsymbol{v})\,\delta p\,\mathrm{d}V = 0
            """
            return ufl.div(J*ufl.inv(F)*v) * var_p * ddomain
        else:
            raise ValueError("Unknown fluid mass formulation!")

    # CH-NS part for reduced mass formulation
    def deltaW_int_pres_reduced_ch(self, alpha, Jflux, var_p, ddomain, F=None):
        J = ufl.det(F)
        """ TeX:
        \int\limits_{\mathit{\Omega}_0}(\alpha\,\widehat{J}\boldsymbol{F}^{-1}\boldsymbol{J}\cdot\nabla\delta p)\,\mathrm{d}V = 0
        """
        return J*ufl.dot(alpha*ufl.inv(F)*Jflux, ufl.grad(var_p)) * ddomain

    def res_v_strong_navierstokes_transient(self, a, v, rho, sig, fbody, w=None, F=None, phi=[None,None], phidot=None):
        return self.f_inert_strong_navierstokes_transient(a, v, rho, w=w, F=F, phi=phi, phidot=phidot) - self.f_stress_strong(sig, F=F) - fbody

    def res_v_strong_navierstokes_steady(self, v, rho, sig, fbody, w=None, F=None, phi=[None,None], phidot=None):
        return self.f_inert_strong_navierstokes_steady(v, rho, w=w, F=F, phi=phi, phidot=phidot) - self.f_stress_strong(sig, F=F) - fbody

    def res_v_strong_stokes_transient(self, a, v, rho, sig, fbody, w=None, F=None, phi=[None,None], phidot=None):
        return self.f_inert_strong_stokes_transient(a, v, rho, w=w, F=F, phi=phi, phidot=phidot) - self.f_stress_strong(sig, F=F) - fbody

    def res_v_strong_stokes_steady(self, rho, sig, fbody, F=None):
        return -self.f_stress_strong(sig, F=F) - fbody

    def f_inert_strong_navierstokes_transient(self, a, v, rho, w=None, F=None, phi=[None,None], phidot=None):
        J = ufl.det(F)
        rho_ = self.get_density(rho, chi=phi[1])
        if self.formulation == "nonconservative":
            assert(phi[0] is None)
            return J*rho_ * (a + ufl.grad(v) * ufl.inv(F) * (v - w))
        elif self.formulation == "conservative":
            Jdot = ufl.div(J*ufl.inv(F)*w)
            if phi[0] is not None:
                rhodot_ = ufl.diff(rho_,phi[0]) * phidot
            else:
                rhodot_ = ufl.as_ufl(0)
            return (J*rhodot_ + rho_*Jdot)*v + J*rho_*a + ufl.div(J*rho_*ufl.outer(v, v - w)*ufl.inv(F).T)
        else:
            raise ValueError("Unknown fluid formulation!")

    def f_inert_strong_navierstokes_steady(self, v, rho, w=None, F=None, phi=[None,None], phidot=None):
        J = ufl.det(F)
        rho_ = self.get_density(rho, chi=phi[1])
        if self.formulation == "nonconservative":
            assert(phi[0] is None)
            return J*rho_ * (ufl.grad(v) * ufl.inv(F) * v)  # NOTE: No domain velocity here! ... Really?!
        elif self.formulation == "conservative":
            return ufl.div(J*rho_*ufl.outer(v, v)*ufl.inv(F).T)  # NOTE: No domain velocity here! ... Really?!
        else:
            raise ValueError("Unknown fluid formulation!")

    def f_inert_strong_stokes_transient(self, a, v, rho, w=None, F=None, phi=[None,None], phidot=None):
        J = ufl.det(F)
        rho_ = self.get_density(rho, chi=phi[1])
        if self.formulation == "nonconservative":
            assert(phi[0] is None)
            return J*rho_ * (a + ufl.grad(v) * ufl.inv(F) * (-w))
        elif self.formulation == "conservative":
            Jdot = ufl.div(J*ufl.inv(F)*w)
            if phi[0] is not None:
                rhodot_ = ufl.diff(rho_,phi[0]) * phidot
            else:
                rhodot_ = ufl.as_ufl(0)
            return (J*rhodot_ + rho_*Jdot)*v + J*rho_*a + ufl.div(J*rho_*ufl.outer(v, -w)*ufl.inv(F).T)
        else:
            raise ValueError("Unknown fluid formulation!")

    def f_stress_strong(self, sig, F=None):
        J = ufl.det(F)
        return ufl.div(J*sig*ufl.inv(F).T)

    def f_gradp_strong(self, p, F=None):
        J = ufl.det(F)
        return J*ufl.inv(F).T * ufl.grad(p)

    # NOTE: For grad/div stabilization, we only use the "reduced mass" residual version here! However, needs investigation...!
    def res_p_strong(self, v, rho, w=None, F=None, phi=[None,None], phidot=None):
        J = ufl.det(F)
        if self.mass_formulation=="conservative_mass":
            rho_ = self.get_density(rho, chi=phi[1])
            Jdot = ufl.div(J*ufl.inv(F)*w)
            if phi[0] is not None:
                rhodot_ = ufl.diff(rho_,phi[0]) * phidot
            else:
                rhodot_ = ufl.as_ufl(0)
            if self.formulation == "nonconservative":
                # return J*rhodot_ + rho_*Jdot + J*ufl.dot(ufl.inv(F).T*ufl.grad(rho_), v-w) + rho_*ufl.div(J*ufl.inv(F)*(v-w))
                return rho_*ufl.div(J*ufl.inv(F)*v)
            elif self.formulation == "conservative":
                # return J*rhodot_ + rho_*Jdot + ufl.div(J*ufl.inv(F)*rho_*(v-w))
                return rho_*ufl.div(J*ufl.inv(F)*v)
            else:
                raise ValueError("Unknown fluid formulation!")
        elif self.mass_formulation=="reduced_mass":
            return ufl.div(J*ufl.inv(F)*v)
        else:
            raise ValueError("Unknown fluid mass formulation!")

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
        vel = v - w # streamline direction should be relavive velocity in ALE
        # NOTE: J=det(F) already included in res_v_strong
        if symmetric:  # modification to make the effective stress symmetric - experimental, use with care...
            return ufl.dot(tau_supg * ufl.sym(ufl.grad(self.var_v) * ufl.inv(F)) * vel, res_v_strong) * ddomain
        else:
            return ufl.dot(tau_supg * ufl.grad(self.var_v) * ufl.inv(F) * vel, res_v_strong) * ddomain

    def stab_pspg(self, var_p, res_v_strong, tau_pspg, rho, ddomain, F=None, phi=[None,None]):
        # NOTE: J=det(F) already included in res_v_strong
        if self.mass_formulation=="conservative_mass":
            return ufl.dot(tau_pspg * ufl.inv(F).T * ufl.grad(var_p), res_v_strong) * ddomain
        elif self.mass_formulation=="reduced_mass":
            rho_ = self.get_density(rho, chi=phi[1])
            return (1./rho_)*ufl.dot(tau_pspg * ufl.inv(F).T * ufl.grad(var_p), res_v_strong) * ddomain
        else:
            raise ValueError("Unknown fluid mass formulation!")

    def stab_lsic(self, v, tau_lsic, rho, ddomain, w=None, F=None, phi=[None,None], phidot=None):
        # NOTE: J=det(F) already included in res_p_strong
        if self.mass_formulation=="conservative_mass":
            return tau_lsic * ufl.inner(ufl.grad(self.var_v), ufl.inv(F).T) * self.res_p_strong(v, rho, w=w, F=F, phi=phi, phidot=phidot) * ddomain
        elif self.mass_formulation=="reduced_mass":
            # rho_ = self.get_density(rho, chi=phi[1])
            rho_ = sum(rho) / len(rho)  # let's use the average here so far...
            return tau_lsic * ufl.inner(ufl.grad(self.var_v), ufl.inv(F).T) * rho_ * self.res_p_strong(v, rho, w=w, F=F, phi=phi, phidot=phidot) * ddomain
        else:
            raise ValueError("Unknown fluid mass formulation!")

    # components of element-level Reynolds number - not used so far... need to assemble a cell-based vector in order to evaluate these!
    def re_c(self, rho, v, ddomain, w=None, F=None, phi=[None,None], phidot=None):
        rho_ = self.get_density(rho, chi=phi[1])
        J = ufl.det(F)
        return rho_ * ufl.dot(ufl.grad(v) * ufl.inv(F) * (v - w), self.var_v) * J * ddomain

    def re_ktilde(self, rho, v, ddomain, w=None, F=None, phi=[None,None], phidot=None):
        rho_ = self.get_density(rho, chi=phi[1])
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
    # TeX: \int\limits_{\mathit{\Gamma}} (\boldsymbol{v}-\widehat{\boldsymbol{w}})\cdot\boldsymbol{n}\,\mathrm{d}a =
    #      \int\limits_{\mathit{\Gamma}_0} (\boldsymbol{v}-\widehat{\boldsymbol{w}})\cdot J\boldsymbol{F}^{-\mathrm{T}}\boldsymbol{n}_0\,\mathrm{d}A
    def flux(self, v, dboundary, w=None, F=None, fcts=None):
        J = ufl.det(F)
        if fcts is None:
            return J * ufl.dot(ufl.inv(F).T * self.n0, (v - w)) * dboundary
        else:
            return (J * ufl.dot(ufl.inv(F).T * self.n0, (v - w)))(fcts) * dboundary

    # capillary force in multiphase flow
    def capillary_force(self, phi, mu, ddomain, F=None, return_type="work"):
        """ TeX:
        \int\limits_{\mathit{\Omega}_0} \widehat{J}\phi\widehat{\boldsymbol{F}}^{-\mathrm{T}}\nabla_{0}\mu \cdot \delta\boldsymbol{v} \,\mathrm{d}V
        """
        J = ufl.det(F)
        return J * ufl.dot(phi * ufl.inv(F).T*ufl.grad(mu), self.var_v) * ddomain

    # Generalized form of Korteweg stress in multiphase flow
    def korteweg_stress(self, phi, mu, psi, kappa, ddomain, F=None, return_type="work"):
        """ TeX:
        \int\limits_{\mathit{\Omega}_0} \widehat{J}\left(\kappa\widehat{\boldsymbol{F}}^{-\mathrm{T}}\nabla_0\phi\otimes\widehat{\boldsymbol{F}}^{-\mathrm{T}}\nabla_0\phi + \left(\mu\phi - \psi -\frac{1}{2}\kappa\widehat{\boldsymbol{F}}^{-\mathrm{T}}\nabla_0\phi\cdot\widehat{\boldsymbol{F}}^{-\mathrm{T}}\nabla_0\phi\right)\boldsymbol{I}\right)\widehat{\boldsymbol{F}}^{-\mathrm{T}} : \nabla_0\delta\boldsymbol{v}\,\mathrm{d}V
        """
        J = ufl.det(F)
        return J * ufl.inner( (kappa*ufl.outer(ufl.inv(F).T*ufl.grad(phi), ufl.inv(F).T*ufl.grad(phi)) + (mu*phi - psi - 0.5*kappa*ufl.dot(ufl.inv(F).T*ufl.grad(phi), ufl.inv(F).T*ufl.grad(phi)))*self.I), ufl.grad(self.var_v)*ufl.inv(F)) * ddomain
