#!/usr/bin/env python3

# Copyright (c) 2019-2026, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl
from ..variationalform import variationalform_base

"""
Scalar transport variational forms class
"""

class variationalform(variationalform_base):
    def __init__(
        self,
        tstfncs=None,
        trlfncs=None,
        n0=None,
        x_ref=None,
        ro0=None,
    ):
        self.var_c = tstfncs[0]
        variationalform_base.__init__(self, tstfncs=tstfncs, n0=n0, x_ref=x_ref, ro0=ro0)

    def diffusion_rate(self, cdot, c, ddomain, v=None, w=None, F=None):
        # advection term if coupled to fluid flow
        if v is not None:
            # NOTE: We should use the conservative form, NOT "ufl.dot(v, ufl.grad(c))"
            advec = ufl.div(c*v)
        else:
            advec = ufl.as_ufl(0)

        return ( ufl.inner(cdot, self.var_c) + ufl.inner(advec, self.var_c) ) * ddomain

    def diffusion_flux(self, difflux, ddomain, F=None):
        return ufl.inner(difflux, ufl.grad(self.var_c)) * ddomain

    def weakform_neumann(self, k, dboundary, F=None):
        return k * self.var_c * dboundary

    def weakform_robin(self, k, c, c0, dboundary, F=None):
        return (k * (c - c0) * self.var_c) * dboundary

    def source_term(self, func, ddomain, F=None, return_type="weak"):
        if return_type=="weak":
            return ufl.dot(func, self.var_c) * ddomain
        elif return_type=="strong":
            return func
        else:
            raise ValueError("Unknown return type.")


class variationalform_ale(variationalform):
    def diffusion_rate(self, cdot, c, ddomain, v=None, w=None, F=None):
        J = ufl.det(F)
        if v is not None:  # up to now, only use if we have a fluid velocity!
            Jdot = ufl.div(J*ufl.inv(F)*w)
        else:
            Jdot = ufl.as_ufl(0)
        # advection term if coupled to fluid flow
        if v is not None:
            # NOTE: We should use the conservative form, NOT "ufl.dot(v-w, ufl.inv(F).T*ufl.grad(c))"
            advec = ufl.div(J*ufl.inv(F)*c*(v-w))
        else:
            advec = ufl.as_ufl(0)

        return ( ufl.inner(J*cdot + c*Jdot, self.var_c) + ufl.inner(advec, self.var_c) ) * ddomain

    def diffusion_flux(self, difflux, ddomain, F=None):
        J = ufl.det(F)
        return J*ufl.inner(ufl.inv(F)*difflux, ufl.grad(self.var_c)) * ddomain

    def source_term(self, func, ddomain, F=None, return_type="weak"):
        J = ufl.det(F)
        if return_type=="weak":
            return J*ufl.dot(func, self.var_c) * ddomain
        elif return_type=="strong":
            return J*func
        else:
            raise ValueError("Unknown return type.")
