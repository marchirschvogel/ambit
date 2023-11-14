#!/usr/bin/env python3

"""
A fluid mechanics problem of incompressible Navier-Stokes flow through a pipe, which is separated by an internal valve (always closed). A 0D model of two 2-element Windkessels in series
bypasses flow from the one region to the other. The setup is rather academic but should illustrate how 0D models can be used to connect independent fluid regions.
"""

import ambit_fe
import numpy as np
from pathlib import Path


def main():

    basepath = str(Path(__file__).parent.absolute())


    """
    Parameters for input/output
    """
    IO_PARAMS            = {# problem type 'fluid_flow0d' indicates a coupling of the individual problems 'fluid' and 'flow0d'
                            'problem_type'          : 'fluid_flow0d',
                            # For this setup in order to model an internal valve/Dirichlet boundary, we need to duplicate the pressure space at this boundary; otherwise,
                            # any Dirichlet condition on the velocity inside the domain will not prevent fluid to experience a pressure gradient across that plane, leading
                            # to unphysical de- and acceleration of fluid infront of and behind the valve.
                            # This duplicate pressure space can only be achieved using the mixed Dolfinx branch, which is installed in the Ambit devenv Docker container.
                            # In the future, this functionality is expected to be merged into the main branch of Dolfinx (as it has been announced).
                            'USE_MIXED_DOLFINX_BRANCH' : True,
                            # indicate which domain IDs (specified in 'mesh_domain' file) should be split (creating a submesh)
                            'duplicate_mesh_domains': [1,2],
                            # the meshes for the domain and boundary topology are specified separately
                            'mesh_domain'           : basepath+'/input/pipe_domain.xdmf',
                            'mesh_boundary'         : basepath+'/input/pipe_boundary.xdmf',
                            'order_fib_input'       : 1,
                            # at which step frequency to write results (set to 0 in order to not write any output)
                            'write_results_every'   : 1,
                            # where to write the output to
                            'output_path'           : basepath+'/tmp/',
                            # which results to write: here, all 3D fields need to be specified, while the 0D model results are output nevertheless
                            'results_to_write'      : ['velocity','pressure'],
                            # the 'midfix' for all simulation result file names: will be results_<simname>_<field>.xdmf/.h5/.txt
                            'simname'               : 'fluid_flow0d_pipe'}
                      
    """
    Parameters for the linear and nonlinear solution schemes
    """
    SOLVER_PARAMS        = {# this specifies which linear solution strategy to use; since this problem has less than 10'000 degrees of freedom, we comfortably can use a direct solver
                            'solve_type'            : 'direct',
                            # residual and increment tolerances: first value for the fluid mechanics problem (momentum balance and continuity), second value for the 3D-0D constraint problem
                            'tol_res'               : [1e-8, 1e-8, 1e-6],
                            'tol_inc'               : [1e-8, 1e-8, 1e-6],
                            # subsolver tolerances for the 0D model solver: tolerances for residual and increment
                            'subsolver_params'      : {'tol_res' : 1e-6,
                                                       'tol_inc' : 1e-6}}

    """
    Parameters for the solid mechanics time integration scheme, plus the global time parameters
    """
    TIME_PARAMS_FLUID    = {# the maximum simulation time - here 0.2 seconds
                            'maxtime'               : 0.2,
                            # the number of time steps which the simulation time is divided into - so here, a time step lasts 0.2/100 s = 0.002 s = 2 ms
                            'numstep'               : 100,
                            # the fluid mechanics time integration scheme: we use a One-Step-theta scheme with theta=1, hence a Backward Euler
                            'timint'                : 'ost',
                            'theta_ost'             : 1.0}

    """
    Parameters for the 0D model time integration scheme
    """
    TIME_PARAMS_FLOW0D   = {# the 0D model time integration scheme: we use a One-Step-theta method with theta = 0.5, which corresponds to the trapezoidal rule
                            'timint'                : 'ost',
                            'theta_ost'             : 0.5,
                            # do initial time step using backward scheme (theta=1), to avoid fluctuations for quantities whose d/dt is zero
                            'initial_backwardeuler' : True,
                            # the initial conditions of the 0D ODE model
                            'initial_conditions'    : {'q_in_0' : 0.0,   # initial in-flux
                                                       'q_d_0' : 0.0,    # initial distal flux
                                                       'p_d_0' : 0.0,    # initial distal pressure
                                                       'q_out_0' : 0.0}} # initial out-flux

    """
    Parameters for the 0D model
    """
    MODEL_PARAMS_FLOW0D  = {# the type of 0D model: 'CRLinoutlink' refers to two Windkessel models in series (each with compliance C, resistance R, and inertance L),
                            # connecting an in- to an out-flow (note that the IN-flow into the 0D model is provided by the OUT-flow of the fluid domain)
                            'modeltype'             : 'CRLinoutlink',
                            # the parameters of the 0D model
                            'parameters'            : {'C_in' : 1e3,    # in-flow compliance (proximal to fluid out-flow surface)
                                                       'R_in' : 160e-6, # in-flow resistance (proximal to fluid out-flow surface)
                                                       'L_in' : 0.0,    # in-flow inertance (proximal to fluid out-flow surface)
                                                       'C_out' : 0.01,  # out-flow compliance (proximal to fluid in-flow surface)
                                                       'R_out' : 1e-6,  # out-flow resistance (proximal to fluid in-flow surface)
                                                       'L_out' : 0.0}}  # out-flow inertance (proximal to fluid in-flow surface)
    """
    Finite element parameters
    """
    FEM_PARAMS           = {# the order of the finite element ansatz functions for the velocity and pressure
                            'order_vel'             : 1,
                            'order_pres'            : 1,
                            # the quadrature degree
                            'quad_degree'           : 5,
                            # stabilization scheme: we make use of a reduced scheme ('supg_pspg2') optimized for first-order elements, which does not have
                            # the inertia term in the strong residual
                            'stabilization'         : {'scheme' : 'supg_pspg2', # scheme name
                                                       'vscale' : 1e3,          # velocity scale
                                                       'dscales' : [1.,1.,1.],  # stabilization parameter scales
                                                       'symmetric' : True}}     # modification to make the effective stress symmetric

    """
    3D-0D coupling parameters
    """
    COUPLING_PARAMS      = {# the surfaces IDs which couple to the 0D world
                            'surface_ids'           : [[4],[6]],
                            # the coupling type: 'monolithic_lagrange' here is a more general scheme that enforces equality of 3D and 0D fluxes/volumes via (Lagrange) multipliers, outsourcing the 0D
                            # solve to a sub-solver in each nonlinear iteration, whereas 'monolithic_direct' would embed the 0D system of equations directly into the monolithic system matrix (which
                            # would be only available for solid mechanics)
                            'coupling_type'         : 'monolithic_lagrange',
                            # for 'monolithic_lagrange', we need the pressures to be the exchanged quantities between 3D and 0D world (being the Lagrange multipliers)
                            'coupling_quantity'     : ['pressure','pressure'],
                            # the coupling variables which are enforced between 3D and 0D: can be volume or flux
                            'variable_quantity'     : ['flux','flux'],
                            # flux scaling: in the 0D world, flux variables are always positively defined, while an in-flux into the fluid domain is against the unit outward surface normal;
                            # so, we need to scale in-fluxes by -1
                            'cq_factor'             : [1.,-1.]}

    """
    Constitutive parameters for the 3D fluid: we have two separate domains, but assign the same parameters to them
    """
    MATERIALS_FLUID      = { # newtonian fluids with parameters for blood - kg-mm-s system
                             'MAT1' : {'newtonian' : {'mu' : 4.0e-6},   # kPa s
                                       'inertia' : {'rho' : 1.025e-6}}, # kg/mm^3
                             'MAT2' : {'newtonian' : {'mu' : 4.0e-6},   # kPa s
                                       'inertia' : {'rho' : 1.025e-6}}} # kg/mm^3


    """
    Time curves, e.g. any prescribed time-controlled/-varying loads or functions
    """
    class time_curves():
        
        def tc1(self, t):
            vmax = 1e3 # mm/s
            T=0.4 # s
            return 0.5*vmax*(1.-np.cos(2.*np.pi*t/T))


    """
    Boundary conditions: ids: 1,5: lateral wall - 2: inflow, 5: axial outflow, 6: top outflow, 3: valve
    Inflow is prescribed, lateral wall as well as "valve" (inner domain separator plane) are fixed, a stabilized Neumann condition is applied to the outflow.
    """
    BC_DICT              = { 'dirichlet' : [{'id' : [2], 'dir' : 'z', 'curve' : 1}, # inflow
                                            {'id' : [1,5], 'dir' : 'all', 'val' : 0.}, # lateral wall
                                            {'id' : [3], 'dir' : 'all', 'val' : 0.}], # inner (valve) plane
                             'stabilized_neumann' : [{'id' : [4,6,7], 'par1' : 0.205e-6, 'par2' : 1.}] }  # par1 should be ~ 0.2*rho

    # Pass parameters to Ambit to set up the problem
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, [TIME_PARAMS_FLUID, TIME_PARAMS_FLOW0D], SOLVER_PARAMS, FEM_PARAMS, [MATERIALS_FLUID, MODEL_PARAMS_FLOW0D], BC_DICT, time_curves=time_curves(), coupling_params=COUPLING_PARAMS)

    # Call the Ambit solver to solve the problem
    problem.solve_problem()



if __name__ == "__main__":

    main()
