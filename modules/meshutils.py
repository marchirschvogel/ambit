#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys, copy
from dolfinx import fem


def gather_surface_dof_indices(io, Vspace, surflist, comm):

    # get boundary dofs which should be reduced
    fn=[]
    for i in range(len(surflist)):

        # these are local node indices!
        fnode_indices_local = fem.locate_dofs_topological(Vspace, io.mesh.topology.dim-1, io.mt_b1.indices[io.mt_b1.values == surflist[i]])

        # get global indices
        fnode_indices = Vspace.dofmap.index_map.local_to_global(fnode_indices_local)

        # gather indices
        fnode_indices_gathered = comm.allgather(fnode_indices)

        # flatten indices from all the processes
        fnode_indices_flat = [item for sublist in fnode_indices_gathered for item in sublist]

        # remove duplicates
        fnode_indices_unique = list(dict.fromkeys(fnode_indices_flat))

        fn.append(fnode_indices_unique)

    # flatten list
    fn_flat = [item for sublist in fn for item in sublist]

    # remove duplicates
    fn_unique = list(dict.fromkeys(fn_flat))

    # now make list of dof indices according to block size
    fd=[]
    for i in range(len(fn_unique)):
        for j in range(Vspace.dofmap.index_map_bs):
            fd.append(Vspace.dofmap.index_map_bs*fn_unique[i]+j)

    return fd
