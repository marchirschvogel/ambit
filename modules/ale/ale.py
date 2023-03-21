#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time, sys
import numpy as np
from dolfinx import fem
import ufl
from petsc4py import PETSc

import ioroutines
import ale_kinematics_constitutive
import ale_variationalform
import timeintegration
import utilities
import solver_nonlin
import boundaryconditions

from base import problem_base, solver_base


# Arbitrary Lagrangian Eulerian (ALE) mechanics problem

class AleProblem(problem_base):

    def __init__(self, io_params, time_params, fem_params, constitutive_models, bc_dict, time_curves, io, mor_params={}, comm=None):
        problem_base.__init__(self, io_params, time_params, comm)

        self.problem_physics = 'ale'
        
        self.simname = io_params['simname']
        self.results_to_write = io_params['results_to_write']
        
        self.io = io

        # number of distinct domains (each one has to be assigned a own material model)
        self.num_domains = len(constitutive_models)
        
        self.constitutive_models = utilities.mat_params_to_dolfinx_constant(constitutive_models, self.io.mesh)

        try: self.order_disp = fem_params['order_disp']
        except: self.order_disp = fem_params['order_vel'] # from fluid problem!
        self.quad_degree = fem_params['quad_degree']
        
        # collect domain data
        self.dx_ = []
        for n in range(self.num_domains):
            # integration domains
            self.dx_.append(ufl.dx(subdomain_data=self.io.mt_d, subdomain_id=n+1, metadata={'quadrature_degree': self.quad_degree}))
        
        # whether to enforce continuity of mass at midpoint or not
        try: self.pressure_at_midpoint = fem_params['pressure_at_midpoint']
        except: self.pressure_at_midpoint = False
        
        self.localsolve = False # no idea what might have to be solved locally...
        self.prestress_initial = False # guess prestressing in ALE is somehow senseless...
        self.incompressible_2field = False # always False here...

        self.sub_solve = False

        self.dim = self.io.mesh.geometry.dim
    
        # type of discontinuous function spaces
        if str(self.io.mesh.ufl_cell()) == 'tetrahedron' or str(self.io.mesh.ufl_cell()) == 'triangle' or str(self.io.mesh.ufl_cell()) == 'triangle3D':
            dg_type = "DG"
            if (self.order_disp > 1) and self.quad_degree < 3:
                raise ValueError("Use at least a quadrature degree of 3 or more for higher-order meshes!")
        elif str(self.io.mesh.ufl_cell()) == 'hexahedron' or str(self.io.mesh.ufl_cell()) == 'quadrilateral' or str(self.io.mesh.ufl_cell()) == 'quadrilateral3D':
            dg_type = "DQ"
            if (self.order_disp > 1) and self.quad_degree < 5:
                raise ValueError("Use at least a quadrature degree of 5 or more for higher-order meshes!")
        else:
            raise NameError("Unknown cell/element type!")

        self.Vex = self.io.mesh.ufl_domain().ufl_coordinate_element()

        # make sure that we use the correct velocity order in case of a higher-order mesh
        if self.Vex.degree() > 1:
            if self.Vex.degree() != self.order_disp:
                raise ValueError("Order of velocity field not compatible with degree of finite element!")

        # check if we want to use model order reduction and if yes, initialize MOR class
        try: self.have_rom = io_params['use_model_order_red']
        except: self.have_rom = False

        if self.have_rom:
            import mor
            self.rom = mor.ModelOrderReduction(mor_params, self.comm)

        # create finite element objects
        P_w = ufl.VectorElement("CG", self.io.mesh.ufl_cell(), self.order_disp)
        # function space
        self.V_w = fem.FunctionSpace(self.io.mesh, P_w)
        # tensor finite element and function space
        P_tensor = ufl.TensorElement("CG", self.io.mesh.ufl_cell(), self.order_disp)
        self.V_tensor = fem.FunctionSpace(self.io.mesh, P_tensor)

        # a discontinuous tensor, vector, and scalar function space
        self.Vd_tensor = fem.TensorFunctionSpace(self.io.mesh, (dg_type, self.order_disp-1))
        self.Vd_vector = fem.VectorFunctionSpace(self.io.mesh, (dg_type, self.order_disp-1))
        self.Vd_scalar = fem.FunctionSpace(self.io.mesh, (dg_type, self.order_disp-1))

        # coordinate element function space
        self.Vcoord = fem.FunctionSpace(self.io.mesh, self.Vex)

        # functions
        self.dw    = ufl.TrialFunction(self.V_w)            # Incremental displacement
        self.var_w = ufl.TestFunction(self.V_w)             # Test function
        self.w     = fem.Function(self.V_w, name="AleDisplacement")
        # old state
        self.w_old = fem.Function(self.V_w)

        self.numdof = self.w.vector.getSize()

        # initialize ALE time-integration class
        self.ti = timeintegration.timeintegration_ale(time_params, time_curves, self.t_init, comm=self.comm)
        
        # initialize kinematics_constitutive class
        self.ki = ale_kinematics_constitutive.kinematics(self.dim)
        
        # initialize material/constitutive classes (one per domain)
        self.ma = []
        for n in range(self.num_domains):
            self.ma.append(ale_kinematics_constitutive.constitutive(self.ki, self.constitutive_models['MAT'+str(n+1)], self.io.mesh))
        
        # initialize ALE variational form class
        self.vf = ale_variationalform.variationalform(self.var_w, self.io.n0)

        # initialize boundary condition class
        self.bc = boundaryconditions.boundary_cond_ale(bc_dict, fem_params, self.io, self.vf, self.ti)
        
        self.bc_dict = bc_dict

        # Dirichlet boundary conditions
        if 'dirichlet' in self.bc_dict.keys():
            self.bc.dirichlet_bcs(self.V_w)

        self.set_variational_forms_and_jacobians()
            
            
    def get_problem_var_list(self):

        is_ghosted = [True]
        return [self.w.vector], is_ghosted
            

    # the main function that defines the fluid mechanics problem in terms of symbolic residual and jacobian forms
    def set_variational_forms_and_jacobians(self):

        # internal virtual work
        self.deltaW_int = ufl.as_ufl(0)
        
        for n in range(self.num_domains):
            # internal virtual work
            self.deltaW_int += self.vf.deltaW_int(self.ma[n].stress(self.w), self.dx_[n])
        
        # external virtual work (from Neumann or Robin boundary conditions, body forces, ...)
        w_neumann, w_robin = ufl.as_ufl(0), ufl.as_ufl(0)
        if 'neumann' in self.bc_dict.keys():
            w_neumann = self.bc.neumann_bcs(self.V_w, self.Vd_scalar)
        if 'robin' in self.bc_dict.keys():
            w_robin = self.bc.robin_bcs(self.w)

        self.deltaW_ext = w_neumann + w_robin

        ### full weakforms 
        
        # internal minus external virtual work
        self.weakform_w = self.deltaW_int - self.deltaW_ext
       
        ### Jacobian
        self.jac_ww = ufl.derivative(self.weakform_w, self.w, self.dw)
            

    def set_forms_solver(self):
        pass


    def assemble_residual_stiffness(self, t, subsolver=None):

        # assemble rhs vector - in case of fluid_ale problem, self.bc.dbcs has DBCs from fluid problem appended
        r_w = fem.petsc.assemble_vector(fem.form(self.weakform_w))
        fem.apply_lifting(r_w, [fem.form(self.jac_ww)], [self.bc.dbcs], x0=[self.w.vector], scale=-1.0)
        r_w.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(r_w, self.bc.dbcs, x0=self.w.vector, scale=-1.0)

        # assemble system matrix
        K_ww = fem.petsc.assemble_matrix(fem.form(self.jac_ww), self.bc.dbcs)
        K_ww.assemble()
        
        return [r_w], [[K_ww]]


    # DEPRECATED: This is something we should actually not do! It will mess with gradients we need w.r.t. the reference (e.g. for FrSI)
    # Instead of moving the mesh, we formulate Navier-Stokes w.r.t. a reference state using the ALE kinematics
    def move_mesh(self):
        
        u = fem.Function(self.Vcoord)
        u.interpolate(self.w)
        self.io.mesh.geometry.x[:,:self.dim] += u.x.array.reshape((-1, self.dim))


    ### now the base routines for this problem
                
    def pre_timestep_routines(self):

        # perform Proper Orthogonal Decomposition
        if self.have_rom:
            self.rom.POD(self, self.V_w)

                
    def read_restart(self, sname, N):

        # read restart information
        if self.restart_step > 0:
            self.io.readcheckpoint(self, N)
            self.simname += '_r'+str(N)

    
    def evaluate_initial(self):
        pass


    def write_output_ini(self):
        
        self.io.write_output(self, writemesh=True)


    def get_time_offset(self):
        return 0.


    def evaluate_pre_solve(self, t):

        # set time-dependent functions
        self.ti.set_time_funcs(self.ti.funcs_to_update, self.ti.funcs_to_update_vec, t)
            
            
    def evaluate_post_solve(self, t, N):
        pass


    def set_output_state(self):
        pass

            
    def write_output(self, N, t, mesh=False): 

        self.io.write_output(self, N=N, t=t)

            
    def update(self):
        
        self.ti.update_timestep(self.w, self.w_old)


    def print_to_screen(self):
        pass
    
    
    def induce_state_change(self):
        pass


    def write_restart(self, sname, N):

        self.io.write_restart(self, N)
        
        
    def check_abort(self, t):
        pass



class AleSolver(solver_base):

    def initialize_nonlinear_solver(self):

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear(self.pb, solver_params=self.solver_params)


    def solve_initial_state(self):
        pass


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(t)


    def print_timestep_info(self, N, t, wt):
    
        # print time step info to screen
        self.pb.ti.print_timestep(N, t, self.solnln.sepstring, wt=wt)
