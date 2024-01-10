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

    print_status("#####################################   AMBIT   #######################################", comm)
    print_status("#################### A FEniCS-based cardiovascular physics solver #####################", comm)


def print_problem(ptype, sname, comm, numdof):

        print_problem_header(comm)

        if ptype == 'solid':
            print_status("###################### Welcome to finite strain solid mechanics #######################", comm)

        elif ptype == 'fluid':
            print_status("############### Welcome to incompressible Navier-Stokes fluid mechanics ###############", comm)

        elif ptype == 'ale':
            print_status("############################## Welcome to ALE mechanics ###############################", comm)

        elif ptype == 'fluid_ale':
            print_status("#### Welcome to incompressible Navier-Stokes fluid mechanics in ALE reference frame ###", comm)

        elif ptype == 'fsi':
            print_status("################# Welcome to monolithic Fluid-Solid Interaction (FSI) #################", comm)

        elif ptype == 'fsi_flow0d':
            print_status("################# Welcome to monolithic Fluid-Solid Interaction (FSI) #################", comm)
            print_status("############################# with coupling to 0D flow ################################", comm)

        elif ptype == 'solid_flow0d':
            print_status("########## Welcome to monolithic coupling of 3D solid mechanics and 0D flow ###########", comm)

        elif ptype == 'solid_flow0d_multiscale_gandr':
            print_status("################# Welcome to multiscale growth and remodeling (G & R) #################", comm)
            print_status("############## Small time scale: Monolithic 3D-0D coupled solid-flow0d ################", comm)
            print_status("####################### Large time scale: Static solid G & R ##########################\n", comm)
            return

        elif ptype == 'solid_constraint':
            print_status("############# Welcome to Lagrange multiplier constraint solid mechanics ###############", comm)

        elif ptype == 'fluid_flow0d':
            print_status("########## Welcome to monolithic coupling of 3D fluid mechanics and 0D flow ###########", comm)

        elif ptype == 'fluid_ale_flow0d':
            print_status("######## Welcome to monolithic coupling of 3D ALE fluid mechanics and 0D flow #########", comm)

        elif ptype == 'flow0d':
            print_status("######################### Welcome to lumped-parameter 0D flow #########################", comm)

        elif ptype == 'signet':
            print_status("######################### Welcome to signalling network models ########################", comm)

        else:
            raise NameError("Unknown problem type!")

        print_sep(comm)

        if isinstance(numdof, list):
            print_status("Number of degrees of freedom: %i + %i" % (numdof[0],numdof[1]), comm)
        else:
            print_status("Number of degrees of freedom: %i" % (numdof), comm)
        print_status("Number of cores: %i" % (comm.size), comm)
        print_status("File name: %s" % (sys.argv[0]), comm)
        print_status("Output specifier name: %s" % (sname), comm)

        print_sep(comm)


def print_sep(comm):

    lensep = 87
    print_status("#"*lensep, comm)


# print prestress info
def print_prestress(inst, comm):

    if inst=='start':
        print_status('Started prestressing...', comm)

    if inst=='updt':
        print_status('Performed MULF update...', comm)

    if inst=='end':
        print_status('Finished prestressing.', comm)


def print_status(message, comm, e="\n"):

    if comm.rank == 0:
        print(message, end=e)
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
                        if isinstance(matparams[k1][k2][k3][k4], dict):
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
