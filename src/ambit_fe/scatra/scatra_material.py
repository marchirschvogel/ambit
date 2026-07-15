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

    def mat_diff(self, params, F=None):
        D = params["D"]

        if F is not None:
            grad_c = ufl.inv(F).T*ufl.grad(self.c)
        else:
            grad_c = ufl.grad(self.c)

        return D * grad_c


    def mat_diff_coup(self, params, F=None):
        D, Dc = params["D"], params["Dc"]
        cc = params["cc"]

        # c_eff = ufl.max_value(self.c, 1e-8)

        if F is not None:
            grad_c = ufl.inv(F).T*ufl.grad(self.c)
            grad_cc = ufl.inv(F).T*ufl.grad(self.c_coup[cc])
        else:
            grad_c = ufl.grad(self.c)
            grad_cc = ufl.grad(self.c_coup[cc])

        return D * (grad_c + Dc * grad_cc)
