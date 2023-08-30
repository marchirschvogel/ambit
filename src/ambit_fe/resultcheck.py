#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import numpy as np
from petsc4py import PETSc

from . import mpiroutines
from .mpiroutines import allgather_vec


# check the result of a node (specified by coordinates) in the full parallel (ghosted) dof vector
def results_check_node(u, check_node, u_corr, V, comm, tol=1.0e-6, nm='vec', readtol=1.0e-8):

    success = True

    # block size of vector to check
    bs = u.vector.getBlockSize()

    # computed errors (difference between simulation and expected results)
    errs = np.zeros(bs*len(check_node))

    # dof coordinates
    co = V.tabulate_dof_coordinates()

    # index map
    #im = V.dofmap.index_map.global_indices() # function seems to have gone!
    im = np.asarray(V.dofmap.index_map.local_to_global(np.arange(V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts, dtype=np.int32)), dtype=PETSc.IntType)

    readtolerance = int(-np.log10(readtol))

    # in parallel, dof indices can be ordered differently, so we need to check the position of the node in the
    # re-ordered local co array and then grep out the corresponding dof index from the index map
    dof_indices, dof_indices_gathered = {}, []
    for i in range(len(check_node)):

        ind = np.where((np.round(check_node[i],readtolerance) == np.round(co,readtolerance)).all(axis=1))[0]

        if len(ind): dof_indices[i] = im[ind[0]]

    # gather indices
    dof_indices_gathered = comm.allgather(dof_indices)

    # make a flat and ordered list of indices (may still have duplicates)
    dof_indices_flat = []
    for i in range(len(check_node)):

        for l in range(len(dof_indices_gathered)):
            if i in dof_indices_gathered[l].keys():
                dof_indices_flat.append(dof_indices_gathered[l][i])

    # create unique list (remove duplicates, but keep order)
    dof_indices_unique = list(dict.fromkeys(dof_indices_flat))

    # gather vector to check
    u_sq = allgather_vec(u.vector, comm)

    for i in range(len(check_node)):
        for j in range(bs):
            errs[bs*i+j] = abs(u_sq[bs*dof_indices_unique[i]+j]-u_corr[bs*i+j])
            if errs[bs*i+j] > tol:
                success = False

    for i in range(len(check_node)):
        for j in range(bs):
            if comm.rank == 0:
                print(nm+"[%i]    = %.16E,    CORR = %.16E,    err = %.16E" % (bs*i+j, u_sq[bs*dof_indices_unique[i]+j], u_corr[bs*i+j], errs[bs*i+j]))
                sys.stdout.flush()

    if comm.rank == 0:
        print("Max error: %E" % (max(errs)))
        sys.stdout.flush()

    return success


# return the final success bool
def success_check(succ, comm):

    success = True

    for b in succ:
        if b == False:
            success = False

    if success:

        if comm.rank == 0:
            print("Test passed. :-)")
            sys.stdout.flush()

    else:

        if comm.rank == 0:
            print("!!!Test failed!!!")
            sys.stdout.flush()

    return success


# check the results of a full parallel (non-ghosted) vector
def results_check_vec(vec, vec_corr, comm, tol=1.0e-6):

    success = True

    vec_sq = allgather_vec(vec, comm)

    errs = np.zeros(len(vec_sq))

    for i in range(len(vec_sq)):
        errs[i] = abs(vec_sq[i]-vec_corr[i])
        if errs[i] > tol:
            success = False

    for i in range(len(vec_sq)):
        if comm.rank == 0:
            print("vec[%i]    = %.16E,    CORR = %E,    err = %E" % (i,vec_sq[i], vec_corr[i], errs[i]))
            sys.stdout.flush()

    return success
