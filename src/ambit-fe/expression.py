#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from petsc4py import PETSc

# template expression class
class template:
    def __init__(self):
        self.val = 0.0

    def evaluate(self, x):
        return np.full(x.shape[1], self.val)


# template vector expression class
class template_vector:
    def __init__(self):
        self.val_x = 0.0
        self.val_y = 0.0
        self.val_z = 0.0

    def evaluate(self, x):
        return ( np.full(x.shape[1], self.val_x),
                 np.full(x.shape[1], self.val_y),
                 np.full(x.shape[1], self.val_z) )


# dummy function
class function_dummy:
    def __init__(self, veclist, comm):
        self.vector = PETSc.Vec().createNest(veclist, comm=comm)
        self.vector.assemble()
