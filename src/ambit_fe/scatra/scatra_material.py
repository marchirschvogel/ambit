#!/usr/bin/env python3

# Copyright (c) 2019-2026, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl


class materiallaw:
    def __init__(self, c, cdot):
        self.c = c
        self.cdot = cdot

    def mat_diff(self, params, F=None):
        D = params["D"]

        if F is not None:
            return D * ufl.inv(F).T*ufl.grad(self.c)
        else:
            return D * ufl.grad(self.c)
