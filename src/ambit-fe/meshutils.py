#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

from dolfinx import fem, mesh
import numpy as np


def gather_surface_dof_indices(io, Vspace, surflist, comm):

    # get boundary dofs into a list
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


# cf. https://fenicsproject.discourse.group/t/transfer-meshtags-to-submesh-in-dolfinx/8952/6
def meshtags_parent_to_child(mshtags, childmsh, childmsh_emap, parentmsh, dimentity):

    if dimentity=='domain':
        dim_p = parentmsh.topology.dim
        dim_c = childmsh.topology.dim
    elif dimentity=='boundary':
        dim_p = parentmsh.topology.dim-1
        dim_c = childmsh.topology.dim-1
    else:
        raise ValueError("Unknown dim entity!")

    d_map = parentmsh.topology.index_map(dim_p)
    all_ent = d_map.size_local + d_map.num_ghosts

    # create array with zeros for all entities that are not marked
    all_values = np.zeros(all_ent, dtype=np.int32)
    all_values[mshtags.indices] = mshtags.values

    c_to_e = parentmsh.topology.connectivity(parentmsh.topology.dim, dim_p)

    childmsh.topology.create_entities(dim_c)
    subf_map = childmsh.topology.index_map(dim_c)
    childmsh.topology.create_connectivity(parentmsh.topology.dim, dim_p)
    c_to_e_sub = childmsh.topology.connectivity(parentmsh.topology.dim, dim_p)
    num_sub_ent = subf_map.size_local + subf_map.size_global

    sub_values = np.zeros(num_sub_ent, dtype=np.int32)

    for i, entity in enumerate(childmsh_emap):
        parent_ent = c_to_e.links(entity)
        child_ent = c_to_e_sub.links(i)
        for child, parent in zip(child_ent, parent_ent):
            sub_values[child] = all_values[parent]

    return mesh.meshtags(childmsh, dim_c, np.arange(num_sub_ent, dtype=np.int32), sub_values)
