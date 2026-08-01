#!/usr/bin/env python3

# Copyright (c) 2019-2026, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl


class materiallaw:
    def __init__(self, c, c_coup):
        self.c = c
        self.c_coup = c_coup

    def diffusion_grad_c(self, params, F=None):
        D = params["D"]

        if F is not None:
            grad_c = ufl.inv(F).T*ufl.grad(self.c)
        else:
            grad_c = ufl.grad(self.c)

        return D * grad_c

    def diffusion_grad_ccoup(self, params, F=None):
        D = params["D"]
        cc = params["cc"]

        if F is not None:
            grad_cc = ufl.inv(F).T*ufl.grad(self.c_coup[cc])
        else:
            grad_cc = ufl.grad(self.c_coup[cc])

        return D * grad_cc

    def diffusion_c_grad_ccoup(self, params, F=None):
        D = params["D"]
        cc = params["cc"]

        if F is not None:
            grad_cc = ufl.inv(F).T*ufl.grad(self.c_coup[cc])
        else:
            grad_cc = ufl.grad(self.c_coup[cc])

        return D * self.c * grad_cc
