#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl

# returns the Cauchy stress sigma for different material laws

class materiallaw:

    def __init__(self, gamma, I):
        self.gamma = gamma
        self.I = I


    def newtonian(self, params):

        mu = params['mu'] # dynamic viscosity

        # classical Newtonian fluid
        sigma = 2.*mu*self.gamma

        return sigma
