#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import sys, copy
from dolfinx import fem
import basix


# print header at beginning of simulation
def print_problem_header(comm):

    if comm.rank == 0:
        print("#####################################   AMBIT   #######################################")
        print("#################### A FEniCS-based cardiovascular physics solver #####################")
        sys.stdout.flush()


def print_problem(ptype, sname, comm, numdof):

        print_problem_header(comm)

        if ptype == 'solid':
            if comm.rank == 0:
                print("###################### Welcome to finite strain solid mechanics #######################")
                sys.stdout.flush()

        elif ptype == 'fluid':
            if comm.rank == 0:
                print("############### Welcome to incompressible Navier-Stokes fluid mechanics ###############")
                sys.stdout.flush()

        elif ptype == 'ale':
            if comm.rank == 0:
                print("############################## Welcome to ALE mechanics ###############################")
                sys.stdout.flush()

        elif ptype == 'fluid_ale':
            if comm.rank == 0:
                print("#### Welcome to incompressible Navier-Stokes fluid mechanics in ALE reference frame ###")
                sys.stdout.flush()

        elif ptype == 'fsi':
            if comm.rank == 0:
                print("################# Welcome to monolithic Fluid-Solid Interaction (FSI) #################")
                sys.stdout.flush()

        elif ptype == 'solid_flow0d':
            if comm.rank == 0:
                print("########## Welcome to monolithic coupling of 3D solid mechanics and 0D flow ###########")
                sys.stdout.flush()

        elif ptype == 'solid_flow0d_multiscale_gandr':
            if comm.rank == 0:
                print("################# Welcome to multiscale growth and remodeling (G & R) #################")
                print("############## Small time scale: Monolithic 3D-0D coupled solid-flow0d ################")
                print("####################### Large time scale: Static solid G & R ##########################\n")
                sys.stdout.flush()
                return

        elif ptype == 'solid_constraint':
            if comm.rank == 0:
                print("############# Welcome to Lagrange multiplier constraint solid mechanics ###############")
                sys.stdout.flush()

        elif ptype == 'fluid_flow0d':
            if comm.rank == 0:
                print("########## Welcome to monolithic coupling of 3D fluid mechanics and 0D flow ###########")
                sys.stdout.flush()

        elif ptype == 'fluid_ale_flow0d':
            if comm.rank == 0:
                print("######## Welcome to monolithic coupling of 3D ALE fluid mechanics and 0D flow #########")
                sys.stdout.flush()

        elif ptype == 'flow0d':
            if comm.rank == 0:
                print("######################### Welcome to lumped-parameter 0D flow #########################")
                sys.stdout.flush()

        elif ptype == 'signet':
            if comm.rank == 0:
                print("######################### Welcome to signalling network models ########################")
                sys.stdout.flush()

        else:
            raise NameError("Unknown problem type!")

        print_sep(comm)
        if comm.rank == 0:
            if isinstance(numdof, list):
                print("Number of degrees of freedom: %i + %i" % (numdof[0],numdof[1]))
            else:
                print("Number of degrees of freedom: %i" % (numdof))
            print("Number of cores: %i" % (comm.size))
            print("File name: %s" % (sys.argv[0]))
            print("Output specifier name: %s" % (sname))
            sys.stdout.flush()
        print_sep(comm)


def print_sep(comm):

    lensep = 87
    if comm.rank == 0:
        print("#"*lensep)


# print prestress info
def print_prestress(inst, comm):

    if inst=='start':
        if comm.rank == 0:
            print("Started prestressing...")
            sys.stdout.flush()

    if inst=='updt':
        if comm.rank == 0:
            print("Performed MULF update...")
            sys.stdout.flush()

    if inst=='end':
        if comm.rank == 0:
            print("Finished prestressing.")
            sys.stdout.flush()


# copies material parameters to be represented as a dolfinx constant (avoids re-compilation upon parameter change)
def mat_params_to_dolfinx_constant(matparams, msh):

    matparams_new = copy.deepcopy(matparams)
    for k1 in matparams.keys():
        for k2 in matparams[k1].keys():
            for k3 in matparams[k1][k2].keys():
                if isinstance(matparams[k1][k2][k3], float):
                    matparams_new[k1][k2][k3] = fem.Constant(msh, matparams[k1][k2][k3])
                # submaterial (e.g. used in remodeling)
                if isinstance(matparams[k1][k2][k3], dict):
                    for k4 in matparams[k1][k2][k3].keys():
                        for k5 in matparams[k1][k2][k3][k4].keys():
                            if isinstance(matparams[k1][k2][k3][k4][k5], float):
                                matparams_new[k1][k2][k3][k4][k5] = fem.Constant(msh, matparams[k1][k2][k3][k4][k5])

    return matparams_new


def get_basix_cell_type(ufl_cell_type):

    if str(ufl_cell_type) == 'tetrahedron':
        return basix.CellType.tetrahedron
    elif str(ufl_cell_type) == 'hexahedron':
        return basix.CellType.hexahedron
    elif str(ufl_cell_type) == 'triangle':
        return basix.CellType.triangle
    elif str(ufl_cell_type) == 'triangle3D':
        return basix.CellType.triangle
    elif str(ufl_cell_type) == 'quadrilateral':
        return basix.CellType.quadrilateral
    elif str(ufl_cell_type) == 'quadrilateral3D':
        return basix.CellType.quadrilateral
    else:
        raise ValueError("Check which cell type you have.")
