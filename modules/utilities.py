#!/usr/bin/env python3

import sys
import numpy as np
from petsc4py import PETSc
from dolfinx import Function, VectorFunctionSpace
from projection import project


# print header at beginning of simulation
def print_problem(ptype, numdof, comm):

    if comm.rank == 0:
        print("#####################################   AMBIT   #######################################")
        sys.stdout.flush()
        print("#################### A FEniCS-based cardiovascular physics solver #####################")
        sys.stdout.flush()
        
        if ptype == 'solid':
            print("###################### Welcome to finite strain solid mechanics #######################")
            sys.stdout.flush()
            
        elif ptype == 'fluid':
            print("############### Welcome to incompressible Navier-Stokes fluid mechanics ###############")
            sys.stdout.flush()
        
        elif ptype == 'solid_flow0d':
            print("########## Welcome to monolithic coupling of 3D solid mechanics and 0D flow ###########")
            sys.stdout.flush()
            
        elif ptype == 'solid_flow0d_multiscalegrowthremodeling':
            print("################# Welcome to multiscale growth and remodeling (G & R) #################")
            print("############## Small time scale: Monolithic 3D-0D coupled solid-flow0d ################")
            print("####################### Large time scale: Static solid G & R ##########################")
            sys.stdout.flush()

        elif ptype == 'fluid_flow0d':
            print("########## Welcome to monolithic coupling of 3D fluid mechanics and 0D flow ###########")
            sys.stdout.flush()

        elif ptype == 'flow0d':
            print("######################### Welcome to lumped-parameter 0D flow #########################")
            sys.stdout.flush()
        
        else:
            raise NameError("Unknown problem type!")

            
        print("#######################################################################################")
        sys.stdout.flush()
        
        print("Number of degrees of freedom: %i" % (numdof))
        sys.stdout.flush()
        print("Number of cores: %i" % (comm.size))
        sys.stdout.flush()
        print("File name: %s" % (sys.argv[0]))
        sys.stdout.flush()
        
        print("#######################################################################################")
        sys.stdout.flush()


# print prestress info
def print_prestress(inst, comm):
    
    if inst=='start':
        if comm.rank == 0:
            print("Prestressing solid in one load step...")
            sys.stdout.flush()

    if inst=='end':
        if comm.rank == 0:
            print("Performed MULF update, finished prestressing.")
            sys.stdout.flush()
