#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


# gather an entry of a parallel PETSc vector
def allgather_vec_entry(var, Id, comm):

    var.assemble()
    # ownership range of parallel vector
    vs, ve = var.getOwnershipRange()

    var_tmp, var_all = 0., 0.

    if Id in range(vs,ve): var_tmp = var[Id]

    var_arr = comm.allgather(var_tmp)

    for i in range(len(var_arr)):
        var_all += var_arr[i]

    return var_all


# gather a parallel PETSc vector and store it to a numpy array known by all processes
def allgather_vec(var, comm):

    var.assemble()
    # ownership range of parallel vector
    vs, ve = var.getOwnershipRange()

    var_tmp, var_all = np.zeros(var.getSize()), np.zeros(var.getSize())

    for i in range(vs,ve):
        var_tmp[i] = var[i]

    var_arr = comm.allgather(var_tmp)

    for i in range(len(var_arr)):
        var_all += var_arr[i]

    del var_tmp

    return var_all


# gather a parallel PETSc matrix and store it to a numpy array known by all processes
def allgather_mat(var, comm):

    var.assemble()
    # (row) ownership range of parallel matrix
    mrs, mre = var.getOwnershipRange()

    var_tmp, var_all = np.zeros((var.getSize()[0],var.getSize()[1])), np.zeros((var.getSize()[0],var.getSize()[1]))

    for i in range(mrs,mre):
        var_tmp[i,:] = var[i,:]

    var_arr = comm.allgather(var_tmp)

    for i in range(len(var_arr)):
        var_all += var_arr[i]

    del var_tmp

    return var_all
