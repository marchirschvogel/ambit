#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl


class materiallaw:
    def __init__(self, d, w, F, elem_metrics):
        self.d = d
        self.w = w
        self.F = F
        self.jac_det = elem_metrics["jac_det"]

        self.dim = len(self.d)
        self.I = ufl.Identity(self.dim)

    def diffusion(self, params):
        D = params["D"]
        scale_det = params.get("scale_det", False)
        scale_exp = params.get("scale_exp", 1.0)

        if scale_det:
            fac = (1.0 / self.jac_det) ** scale_exp
        else:
            fac = 1.0

        return fac * D * ufl.grad(self.d)

    def diffusion_rate(self, params):
        D = params["D"]
        scale_det = params.get("scale_det", False)
        scale_exp = params.get("scale_exp", 1.0)

        if scale_det:
            fac = (1.0 / self.jac_det) ** scale_exp
        else:
            fac = 1.0

        return fac * D * ufl.grad(self.w)

    def diffusion_sym(self, params):
        D = params["D"]
        scale_det = params.get("scale_det", False)
        scale_exp = params.get("scale_exp", 1.0)

        if scale_det:
            fac = (1.0 / self.jac_det) ** scale_exp
        else:
            fac = 1.0

        return fac * D * ufl.sym(ufl.grad(self.d))

    def diffusion_rate_sym(self, params):
        D = params["D"]
        scale_det = params.get("scale_det", False)
        scale_exp = params.get("scale_exp", 1.0)

        if scale_det:
            fac = (1.0 / self.jac_det) ** scale_exp
        else:
            fac = 1.0

        return fac * D * ufl.sym(ufl.grad(self.w))

    def linelast(self, params):
        Emod = params["Emod"]
        nu = params["nu"]
        scale_det = params.get("scale_det", False)
        scale_exp = params.get("scale_exp", 1.0)

        if scale_det:
            fac = (1.0 / self.jac_det) ** scale_exp
        else:
            fac = 1.0

        mu = Emod / (2.0 * (1.0 + nu))
        lam = nu * Emod / ((1.0 + nu) * (1.0 - 2.0 * nu))

        epsilon = ufl.sym(ufl.grad(self.d))

        # stress
        return fac * (2.0 * mu * epsilon + lam * ufl.tr(epsilon) * self.I)

    def neohooke(self, params):
        # shear modulus and Poisson's ratio
        mu, nu = params["mu"], params["nu"]
        scale_det = params.get("scale_det", False)
        scale_exp = params.get("scale_exp", 1.0)

        if scale_det:
            fac = (1.0 / self.jac_det) ** scale_exp
        else:
            fac = 1.0

        beta = nu / (1.0 - 2.0 * nu)

        # first invariant and determinant of right Cauchy-Green tensor
        Ic = ufl.tr(self.F.T * self.F)
        J = ufl.det(self.F)

        # compressible NeoHookean material (Holzapfel eq. 6.148)
        Psi = (mu / 2.0) * (Ic - self.dim) + mu / (2.0 * beta) * (J ** (-2.0 * beta) - 1.0)

        # stress
        return fac * ufl.diff(Psi, self.F)

    def exponential(self, params):
        a_0, b_0, kappa = params["a_0"], params["b_0"], params["kappa"]
        scale_det = params.get("scale_det", False)
        scale_exp = params.get("scale_exp", 1.0)

        if scale_det:
            fac = (1.0 / self.jac_det) ** scale_exp
        else:
            fac = 1.0

        J = ufl.det(self.F)
        Ic_bar = J ** (-2.0 / self.dim) * ufl.tr(self.F.T * self.F)

        # exponential law: can be soft in the small deformation realm and stiffer for larger deformations
        Psi = a_0 / (2.0 * b_0) * (ufl.exp(b_0 * (Ic_bar - self.dim)) - 1.0) + (kappa / 2.0) * (J - 1.0) ** 2.0

        # stress
        return fac * ufl.diff(Psi, self.F)
