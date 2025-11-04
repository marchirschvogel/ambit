#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl


class materiallaw:
    def __init__(self, phi):
        self.phi = phi

    def mat_cahnhilliard(self, params):
        D = params["D"]
        psi = D * self.phi**2. * (1. - self.phi) ** 2.

        return ufl.diff(psi,self.phi)
