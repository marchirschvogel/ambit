#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl
from ..variationalform import variationalform_base

"""
ALE variational forms class
Principle of Virtual Work
\delta \mathcal{W} = \delta \mathcal{W}_{\mathrm{int}} - \delta \mathcal{W}_{\mathrm{ext}} = 0, \quad \forall \; \delta\boldsymbol{u}
"""


class variationalform(variationalform_base):
    ### Internal virtual work

    # TeX: \delta \mathcal{W}_{\mathrm{int}} := \int\limits_{\Omega_0} \boldsymbol{\sigma} : \delta\boldsymbol{\epsilon} \,\mathrm{d}V
    def deltaW_int(self, stress, ddomain):
        return ufl.inner(stress, ufl.grad(self.var_d)) * ddomain

    # Robin term for weak imposition of Dirichlet condition
    # TeX: \int\limits_{\Gamma_0} J\beta\,(\boldsymbol{u}-\boldsymbol{u}_{\mathrm{D}})\cdot\delta\boldsymbol{u}\sqrt{\boldsymbol{n}_0 \cdot (\boldsymbol{F}^{-1}\boldsymbol{F}^{-\mathrm{T}})\boldsymbol{n}_0}\,\mathrm{d}A

    # Nitsche term for weak imposition of Dirichlet condition
    # TeX: \int\limits_{\Gamma_0} \frac{\beta}{2 r_{\mathrm{o}}}\,(\boldsymbol{u}-\boldsymbol{u}_{\mathrm{D}})\cdot\delta\boldsymbol{u}\,\mathrm{d}A - \int\limits_{\Gamma_0} \boldsymbol{\sigma}(\delta\boldsymbol{u})\boldsymbol{n}_{0}\cdot (\boldsymbol{u}-\boldsymbol{u}_{\mathrm{D}})\,\mathrm{d}A
    def deltaW_int_nitsche_dirichlet(self, u, uD, var_stress, beta, dboundary, hscale=True):
        if hscale:  # NOTE: Cannot use circumradius for non-simplex cells, so the BC should be called without hscale...
            return (
                (beta / (2.0 * self.ro0)) * ufl.dot((u - uD), self.var_d)
                - ufl.dot(ufl.dot(var_stress, self.n0), (u - uD))
            ) * dboundary
        else:
            return (beta * ufl.dot((u - uD), self.var_d) - ufl.dot(ufl.dot(var_stress, self.n0), (u - uD))) * dboundary
