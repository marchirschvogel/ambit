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
import fluid_kinematics_constitutive
import fluid_variationalform
import timeintegration
import utilities
import solver_nonlin
import boundaryconditions
from projection import project
from solid_material import activestress_activation

from base import problem_base, solver_base

# fluid mechanics, governed by incompressible Navier-Stokes equations:

#\begin{align}
#\rho \left(\frac{\partial\boldsymbol{v}}{\partial t} + \left(\boldsymbol{\nabla}\boldsymbol{v}\right) \boldsymbol{v}\right) = \boldsymbol{\nabla} \cdot \boldsymbol{\sigma} + \hat{\boldsymbol{b}} \quad \text{in} \; \Omega \times [0, T] \\
#\boldsymbol{\nabla} \cdot \boldsymbol{v} = 0 \quad \text{in} \; \Omega \times [0, T]
#\end{align}

class FluidmechanicsProblem(problem_base):

    def __init__(self, io_params, time_params, fem_params, constitutive_models, bc_dict, time_curves, io, mor_params={}, comm=None, alevar={}):
        problem_base.__init__(self, io_params, time_params, comm)

        self.problem_physics = 'fluid'

        self.simname = io_params['simname']
        self.results_to_write = io_params['results_to_write']

        self.io = io

        # number of distinct domains (each one has to be assigned a own material model)
        self.num_domains = len(constitutive_models)

        self.constitutive_models = utilities.mat_params_to_dolfinx_constant(constitutive_models, self.io.mesh)

        self.order_vel = fem_params['order_vel']
        self.order_pres = fem_params['order_pres']
        self.quad_degree = fem_params['quad_degree']

        # collect domain data
        self.dx_, self.rho = [], []
        for n in range(self.num_domains):
            # integration domains
            self.dx_.append(ufl.dx(subdomain_data=self.io.mt_d, subdomain_id=n+1, metadata={'quadrature_degree': self.quad_degree}))
            # data for inertial forces: density
            self.rho.append(self.constitutive_models['MAT'+str(n+1)]['inertia']['rho'])

        # whether to enforce continuity of mass at midpoint or not
        try: self.pressure_at_midpoint = fem_params['pressure_at_midpoint']
        except: self.pressure_at_midpoint = False

        try: self.fluid_formulation = fem_params['fluid_formulation']
        except: self.fluid_formulation = 'nonconservative'

        try: self.fluid_governing_type = time_params['fluid_governing_type']
        except: self.fluid_governing_type = 'navierstokes_transient'

        try: self.stabilization = fem_params['stabilization']
        except: self.stabilization = None

        try: self.prestress_initial = fem_params['prestress_initial']
        except: self.prestress_initial = False
        try: self.prestress_numstep = fem_params['prestress_numstep']
        except: self.prestress_numstep = 1
        try: self.prestress_maxtime = fem_params['prestress_maxtime']
        except: self.prestress_maxtime = 1.0
        try: self.prestress_ptc = fem_params['prestress_ptc']
        except: self.prestress_ptc = False
        try: self.prestress_from_file = fem_params['prestress_from_file']
        except: self.prestress_from_file = False

        if self.prestress_from_file: self.prestress_initial = False

        self.localsolve = False # no idea what might have to be solved locally...
        self.p11 = ufl.as_ufl(0) # can't think of a fluid case with non-zero 11-block in system matrix...

        self.sub_solve = False

        self.dim = self.io.mesh.geometry.dim

        # type of discontinuous function spaces
        if str(self.io.mesh.ufl_cell()) == 'tetrahedron' or str(self.io.mesh.ufl_cell()) == 'triangle' or str(self.io.mesh.ufl_cell()) == 'triangle3D':
            dg_type = "DG"
            if (self.order_vel > 1 or self.order_pres > 1) and self.quad_degree < 3:
                raise ValueError("Use at least a quadrature degree of 3 or more for higher-order meshes!")
        elif str(self.io.mesh.ufl_cell()) == 'hexahedron' or str(self.io.mesh.ufl_cell()) == 'quadrilateral' or str(self.io.mesh.ufl_cell()) == 'quadrilateral3D':
            dg_type = "DQ"
            if (self.order_vel > 1 or self.order_pres > 1) and self.quad_degree < 5:
                raise ValueError("Use at least a quadrature degree of 5 or more for higher-order meshes!")
        else:
            raise NameError("Unknown cell/element type!")

        self.Vex = self.io.mesh.ufl_domain().ufl_coordinate_element()

        if self.stabilization is None and self.order_vel == self.order_pres:
            raise ValueError("Equal order velocity and pressure interpolation is not recommended for non-stabilized Navier-Stokes!")

        # check if we want to use model order reduction and if yes, initialize MOR class
        try: self.have_rom = io_params['use_model_order_red']
        except: self.have_rom = False

        # ALE fluid problem variables
        self.alevar = alevar

        # create finite element objects for v and p
        P_v = ufl.VectorElement("CG", self.io.mesh.ufl_cell(), self.order_vel)
        P_p = ufl.FiniteElement("CG", self.io.mesh.ufl_cell(), self.order_pres)
        # function spaces for v and p
        self.V_v = fem.FunctionSpace(self.io.mesh, P_v)
        self.V_p = fem.FunctionSpace(self.io.mesh, P_p)
        # continuous tensor and scalar function spaces of order order_vel
        self.V_tensor = fem.TensorFunctionSpace(self.io.mesh, ("CG", self.order_vel))
        self.V_scalar = fem.FunctionSpace(self.io.mesh, ("CG", self.order_vel))

        # a discontinuous tensor, vector, and scalar function space
        self.Vd_tensor = fem.TensorFunctionSpace(self.io.mesh, (dg_type, self.order_vel-1))
        self.Vd_vector = fem.VectorFunctionSpace(self.io.mesh, (dg_type, self.order_vel-1))
        self.Vd_scalar = fem.FunctionSpace(self.io.mesh, (dg_type, self.order_vel-1))

        # functions
        self.dv     = ufl.TrialFunction(self.V_v)            # Incremental velocity
        self.var_v  = ufl.TestFunction(self.V_v)             # Test function
        self.dp     = ufl.TrialFunction(self.V_p)            # Incremental pressure
        self.var_p  = ufl.TestFunction(self.V_p)             # Test function
        self.v      = fem.Function(self.V_v, name="Velocity")
        self.p      = fem.Function(self.V_p, name="Pressure")
        # values of previous time step
        self.v_old  = fem.Function(self.V_v)
        self.a_old  = fem.Function(self.V_v)
        self.p_old  = fem.Function(self.V_p)
        # a fluid displacement
        self.uf_old = fem.Function(self.V_v)
        # active stress for reduced solid
        self.tau_a  = fem.Function(self.Vd_scalar, name="tau_a")
        self.tau_a_old = fem.Function(self.Vd_scalar)
        # prestress displacement for FrSI
        if self.prestress_initial or self.prestress_from_file:
            self.uf_pre = fem.Function(self.V_v, name="Displacement_prestress")
        else:
            self.uf_pre = None

        # own read function: requires plain txt format of type valx valy valz x z y
        if self.prestress_from_file:
            self.io.readfunction(self.uf_pre, self.V_v, self.prestress_from_file)

        # dictionaries of internal variables
        self.internalvars, self.internalvars_old = {}, {}

        self.numdof = self.v.vector.getSize() + self.p.vector.getSize()

        if self.have_rom:
            import mor
            self.rom = mor.ModelOrderReduction(mor_params, [self.V_v,self.V_scalar], self.io, self.comm)

        # initialize fluid time-integration class
        self.ti = timeintegration.timeintegration_fluid(time_params, fem_params, time_curves=time_curves, t_init=self.t_init, comm=self.comm)

        # get time factors
        self.timefac_m, self.timefac = self.ti.timefactors()

        # initialize kinematics_constitutive class
        self.ki = fluid_kinematics_constitutive.kinematics(self.dim, uf_pre=self.uf_pre)

        # initialize material/constitutive classes (one per domain)
        self.ma = []
        for n in range(self.num_domains):
            self.ma.append(fluid_kinematics_constitutive.constitutive(self.ki, self.constitutive_models['MAT'+str(n+1)]))

        # initialize fluid variational form class
        if not bool(self.alevar):
            # standard Eulerian fluid
            self.alevar = {'Fale' : None, 'Fale_old' : None, 'w' : None, 'w_old' : None, 'fluid_on_deformed' : 'no'}
            self.vf = fluid_variationalform.variationalform(self.var_v, self.dv, self.var_p, self.dp, self.io.n0, formulation=self.fluid_formulation)

        else:
            # mid-point representation of ALE velocity
            self.alevar['w_mid']    = self.timefac * self.alevar['w']    + (1.-self.timefac) * self.alevar['w_old']
            # mid-point representation of ALE deformation gradient - linear in ALE displacement, hence we can combine it like this
            self.alevar['Fale_mid'] = self.timefac * self.alevar['Fale'] + (1.-self.timefac) * self.alevar['Fale_old']

            if self.alevar['fluid_on_deformed'] == 'consistent':
                # fully consistent ALE formulation of Navier-Stokes
                self.vf = fluid_variationalform.variationalform_ale(self.var_v, self.dv, self.var_p, self.dp, self.io.n0, formulation=self.fluid_formulation)

            elif self.alevar['fluid_on_deformed'] == 'from_last_step':
                # ALE formulation of Navier-Stokes using metrics (Fale, w) from the last converged step... more efficient but not fully consistent
                self.alevar['Fale'], self.alevar['w'] = self.alevar['Fale_old'], self.alevar['w_old']
                self.vf = fluid_variationalform.variationalform_ale(self.var_v, self.dv, self.var_p, self.dp, self.io.n0, formulation=self.fluid_formulation)

            elif self.alevar['fluid_on_deformed'] == 'mesh_move':
                # Navier-Stokes formulated w.r.t. the current, moved frame... more efficient than 'consistent' approach but not fully consistent
                # WARNING: This is unsuitable for FrSI, as we need gradients w.r.t. the reference frame on the reduced boundary!
                self.alevar = {'Fale' : None, 'Fale_old' : None, 'w' : None, 'w_old' : None, 'fluid_on_deformed' : 'mesh_move'}
                self.vf = fluid_variationalform.variationalform(self.var_v, self.dv, self.var_p, self.dp, self.io.n0, formulation=self.fluid_formulation)

            else:
                raise ValueError("Unkown fluid_on_deformed option!")

        # read in fiber data - for reduced solid (FrSI)
        if bool(self.io.fiber_data):

            fibarray = ['circ']
            if len(self.io.fiber_data)>1: fibarray.append('long')

            self.fib_func = self.io.readin_fibers(fibarray, self.V_v, self.dx_, self.order_vel)

            if 'fibers' in self.results_to_write:
                for i in range(len(fibarray)):
                    fib_proj = project(self.fib_func[i], self.V_v, self.dx_, nm='Fiber'+str(i+1))
                    self.io.write_output_pre(self, fib_proj, 0.0, 'fib_'+fibarray[i])

        else:
            self.fib_func = None

        # initialize boundary condition class
        self.bc = boundaryconditions.boundary_cond_fluid(fem_params, self.io, self.vf, self.ti, ki=self.ki, ff=self.fib_func)

        self.bc_dict = bc_dict

        # Dirichlet boundary conditions
        if 'dirichlet' in self.bc_dict.keys():
            self.bc.dirichlet_bcs(self.bc_dict['dirichlet'], self.V_v)

        self.set_variational_forms()


    def get_problem_var_list(self):

        is_ghosted = [True]*2
        return [self.v.vector, self.p.vector], is_ghosted


    # the main function that defines the fluid mechanics problem in terms of symbolic residual and jacobian forms
    def set_variational_forms(self):

        # set form for acceleration
        self.acc = self.ti.set_acc(self.v, self.v_old, self.a_old)
        # set form for fluid displacement (needed for FrSI)
        self.ufluid = self.ti.set_uf(self.v, self.v_old, self.uf_old)

        # set mid-point representations (if needed...)
        self.acc_mid     = self.timefac_m * self.acc    + (1.-self.timefac_m) * self.a_old
        self.vel_mid     = self.timefac   * self.v      + (1.-self.timefac)   * self.v_old
        self.ufluid_mid  = self.timefac   * self.ufluid + (1.-self.timefac)   * self.uf_old

        # kinetic, internal, and pressure virtual power
        self.deltaW_kin, self.deltaW_kin_old = ufl.as_ufl(0), ufl.as_ufl(0)
        self.deltaW_int, self.deltaW_int_old = ufl.as_ufl(0), ufl.as_ufl(0)
        self.deltaW_p,   self.deltaW_p_old   = ufl.as_ufl(0), ufl.as_ufl(0)

        for n in range(self.num_domains):

            # kinetic virtual power
            if self.fluid_governing_type=='navierstokes_transient':
                self.deltaW_kin     += self.vf.deltaW_kin_navierstokes_transient(self.acc, self.v, self.rho[n], self.dx_[n], w=self.alevar['w'], Fale=self.alevar['Fale'])
                self.deltaW_kin_old += self.vf.deltaW_kin_navierstokes_transient(self.a_old, self.v_old, self.rho[n], self.dx_[n], w=self.alevar['w_old'], Fale=self.alevar['Fale_old'])
            elif self.fluid_governing_type=='navierstokes_steady':
                self.deltaW_kin     += self.vf.deltaW_kin_navierstokes_steady(self.v, self.rho[n], self.dx_[n], w=self.alevar['w'], Fale=self.alevar['Fale'])
                self.deltaW_kin_old += self.vf.deltaW_kin_navierstokes_steady(self.v_old, self.rho[n], self.dx_[n], w=self.alevar['w_old'], Fale=self.alevar['Fale_old'])
            elif self.fluid_governing_type=='stokes_transient':
                self.deltaW_kin     += self.vf.deltaW_kin_stokes_transient(self.acc, self.v, self.rho[n], self.dx_[n], w=self.alevar['w'], Fale=self.alevar['Fale'])
                self.deltaW_kin_old += self.vf.deltaW_kin_stokes_transient(self.a_old, self.v_old, self.rho[n], self.dx_[n], w=self.alevar['w_old'], Fale=self.alevar['Fale_old'])
            elif self.fluid_governing_type=='stokes_steady':
                pass # no kinetic term to add for steady Stokes flow
            else:
                raise ValueError("Unknown fluid_governing_type!")

            # internal virtual power
            self.deltaW_int     += self.vf.deltaW_int(self.ma[n].sigma(self.v, self.p, Fale=self.alevar['Fale']), self.dx_[n], Fale=self.alevar['Fale'])
            self.deltaW_int_old += self.vf.deltaW_int(self.ma[n].sigma(self.v_old, self.p_old, Fale=self.alevar['Fale_old']), self.dx_[n], Fale=self.alevar['Fale_old'])

            # pressure virtual power
            self.deltaW_p       += self.vf.deltaW_int_pres(self.v, self.dx_[n], Fale=self.alevar['Fale'])
            self.deltaW_p_old   += self.vf.deltaW_int_pres(self.v_old, self.dx_[n], Fale=self.alevar['Fale_old'])

        # external virtual power (from Neumann or Robin boundary conditions, body forces, ...)
        w_neumann, w_neumann_old, w_robin, w_robin_old, w_stabneumann, w_stabneumann_old, w_membrane, w_membrane_old = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)
        if 'neumann' in self.bc_dict.keys():
            w_neumann     = self.bc.neumann_bcs(self.bc_dict['neumann'], self.V_v, self.Vd_scalar, Fale=self.alevar['Fale'], funcs_to_update=self.ti.funcs_to_update, funcs_to_update_vec=self.ti.funcs_to_update_vec)
            w_neumann_old = self.bc.neumann_bcs(self.bc_dict['neumann'], self.V_v, self.Vd_scalar, Fale=self.alevar['Fale_old'], funcs_to_update=self.ti.funcs_to_update_old, funcs_to_update_vec=self.ti.funcs_to_update_vec_old)
        if 'robin' in self.bc_dict.keys():
            w_robin     = self.bc.robin_bcs(self.bc_dict['robin'], self.v, Fale=self.alevar['Fale'])
            w_robin_old = self.bc.robin_bcs(self.bc_dict['robin'], self.v_old, Fale=self.alevar['Fale_old'])
        if 'stabilized_neumann' in self.bc_dict.keys():
            w_stabneumann     = self.bc.stabilized_neumann_bcs(self.bc_dict['stabilized_neumann'], self.v, wel=self.alevar['w'], Fale=self.alevar['Fale'])
            w_stabneumann_old = self.bc.stabilized_neumann_bcs(self.bc_dict['stabilized_neumann'], self.v_old, wel=self.alevar['w_old'], Fale=self.alevar['Fale_old'])
        # reduced-solid for FrSI problem
        self.have_active_stress, self.active_stress_trig = False, 'ode'
        if 'membrane' in self.bc_dict.keys():
            assert(self.alevar['fluid_on_deformed']!='mesh_move')

            self.mem_active_stress = [False]*len(self.bc_dict['membrane'])

            self.internalvars['tau_a'], self.internalvars_old['tau_a'] = self.tau_a, self.tau_a_old

            self.actstress = []
            for nm in range(len(self.bc_dict['membrane'])):

                if 'active_stress' in self.bc_dict['membrane'][nm]['params'].keys():
                    self.mem_active_stress[nm], self.have_active_stress = True, True

                    act_curve = self.ti.timecurves(self.bc_dict['membrane'][nm]['params']['active_stress']['activation_curve'])
                    self.actstress.append(activestress_activation(self.bc_dict['membrane'][nm]['params']['active_stress'], act_curve))

            w_membrane, self.dbmem, self.bstress = self.bc.membranesurf_bcs(self.bc_dict['membrane'], self.ufluid, self.v, self.acc, self.var_v, ivar=self.internalvars)
            w_membrane_old, _, _                 = self.bc.membranesurf_bcs(self.bc_dict['membrane'], self.uf_old, self.v_old, self.a_old, self.var_v, ivar=self.internalvars_old)

        w_neumann_prestr, self.deltaW_prestr_kin = ufl.as_ufl(0), ufl.as_ufl(0)
        if self.prestress_initial:
            self.funcs_to_update_pre, self.funcs_to_update_vec_pre = [], []
            # Stokes kinetic virtual power
            for n in range(self.num_domains):
                #self.deltaW_prestr_kin += self.vf.deltaW_kin_navierstokes(self.acc, self.v, self.rho[n], self.dx_[n], w=self.alevar['w'], Fale=self.alevar['Fale'])
                self.deltaW_prestr_kin += self.vf.deltaW_kin_stokes_transient(self.acc, self.v, self.rho[n], self.dx_[n], w=self.alevar['w'], Fale=self.alevar['Fale'])
            if 'neumann_prestress' in self.bc_dict.keys():
                w_neumann_prestr = self.bc.neumann_bcs(self.bc_dict['neumann_prestress'], self.V_v, self.Vd_scalar, Fale=self.alevar['Fale'], funcs_to_update=self.funcs_to_update_pre, funcs_to_update_vec=self.funcs_to_update_vec_pre)
            self.deltaW_prestr_ext = w_neumann_prestr + w_robin + w_stabneumann + w_membrane
        else:
            assert('neumann_prestress' not in self.bc_dict.keys())

        # TODO: Body forces!
        self.deltaW_ext     = w_neumann + w_robin + w_stabneumann + w_membrane
        self.deltaW_ext_old = w_neumann_old + w_robin_old + w_stabneumann_old + w_membrane_old

        # stabilization
        if self.stabilization is not None:

            # should only be used for equal-order approximations
            assert(self.order_vel==self.order_pres)

            vscale = self.stabilization['vscale']
            h = self.io.hd0 # cell diameter (could also use max edge length self.io.emax0, but seems to yield similar/same results)

            # full scheme
            if self.stabilization['scheme']=='supg_pspg':

                for n in range(self.num_domains):

                    tau_supg = h / vscale
                    tau_pspg = h / vscale
                    tau_lsic = h * vscale

                    # strong momentum residuals
                    if self.fluid_governing_type=='navierstokes_transient':
                        residual_v_strong     = self.vf.res_v_strong_navierstokes_transient(self.acc, self.v, self.rho[n], self.ma[n].sigma(self.v, self.p, Fale=self.alevar['Fale']), w=self.alevar['w'], Fale=self.alevar['Fale'])
                        residual_v_strong_old = self.vf.res_v_strong_navierstokes_transient(self.a_old, self.v_old, self.rho[n], self.ma[n].sigma(self.v_old, self.p_old, Fale=self.alevar['Fale_old']), w=self.alevar['w_old'], Fale=self.alevar['Fale_old'])
                    elif self.fluid_governing_type=='navierstokes_steady':
                        residual_v_strong     = self.vf.res_v_strong_navierstokes_steady(self.v, self.rho[n], self.ma[n].sigma(self.v, self.p, Fale=self.alevar['Fale']), w=self.alevar['w'], Fale=self.alevar['Fale'])
                        residual_v_strong_old = self.vf.res_v_strong_navierstokes_steady(self.v_old, self.rho[n], self.ma[n].sigma(self.v_old, self.p_old, Fale=self.alevar['Fale_old']), w=self.alevar['w_old'], Fale=self.alevar['Fale_old'])
                    elif self.fluid_governing_type=='stokes_transient':
                        residual_v_strong     = self.vf.res_v_strong_stokes_transient(self.acc, self.v, self.rho[n], self.ma[n].sigma(self.v, self.p, Fale=self.alevar['Fale']), w=self.alevar['w'], Fale=self.alevar['Fale'])
                        residual_v_strong_old = self.vf.res_v_strong_stokes_transient(self.a_old, self.v_old, self.rho[n], self.ma[n].sigma(self.v_old, self.p_old, Fale=self.alevar['Fale_old']), w=self.alevar['w_old'], Fale=self.alevar['Fale_old'])
                    elif self.fluid_governing_type=='stokes_steady':
                        residual_v_strong     = self.vf.res_v_strong_stokes_steady(self.rho[n], self.ma[n].sigma(self.v, self.p, Fale=self.alevar['Fale']), Fale=self.alevar['Fale'])
                        residual_v_strong_old = self.vf.res_v_strong_stokes_steady(self.rho[n], self.ma[n].sigma(self.v_old, self.p_old, Fale=self.alevar['Fale_old']), Fale=self.alevar['Fale_old'])
                    else:
                        raise ValueError("Unknown fluid_governing_type!")

                    # SUPG (streamline-upwind Petrov-Galerkin) for Navier-Stokes
                    if self.fluid_governing_type=='navierstokes_transient' or self.fluid_governing_type=='navierstokes_steady':
                        self.deltaW_int     += self.vf.stab_supg(self.acc, self.v, self.p, residual_v_strong, tau_supg, self.rho[n], self.dx_[n], w=self.alevar['w'], Fale=self.alevar['Fale'])
                        self.deltaW_int_old += self.vf.stab_supg(self.a_old, self.v_old, self.p_old, residual_v_strong_old, tau_supg, self.rho[n], self.dx_[n], w=self.alevar['w_old'], Fale=self.alevar['Fale_old'])
                    # PSPG (pressure-stabilizing Petrov-Galerkin) for Navier-Stokes and Stokes
                    self.deltaW_p       += self.vf.stab_pspg(self.acc, self.v, self.p, residual_v_strong, tau_pspg, self.rho[n], self.dx_[n], Fale=self.alevar['Fale'])
                    self.deltaW_p_old   += self.vf.stab_pspg(self.a_old, self.v_old, self.p_old, residual_v_strong_old, tau_pspg, self.rho[n], self.dx_[n], Fale=self.alevar['Fale_old'])
                    # LSIC (least-squares on incompressibility constraint) for Navier-Stokes and Stokes
                    self.deltaW_int     += self.vf.stab_lsic(self.v, tau_lsic, self.rho[n], self.dx_[n], Fale=self.alevar['Fale'])
                    self.deltaW_int_old += self.vf.stab_lsic(self.v_old, tau_lsic, self.rho[n], self.dx_[n], Fale=self.alevar['Fale_old'])

            # reduced scheme: missing transient NS term as well as divergence stress term of strong residual
            if self.stabilization['scheme']=='supg_pspg2':

                # optimized for first-order
                assert(self.order_vel==1)
                assert(self.order_pres==1)

                for n in range(self.num_domains):

                    dscales = self.stabilization['dscales']

                    # yields same result as above for navierstokes_steady
                    delta1 = dscales[0] * self.rho[n] * h / vscale
                    delta2 = dscales[1] * self.rho[n] * h * vscale
                    delta3 = dscales[2] * h / vscale

                    self.deltaW_int     += self.vf.stab_v(delta1, delta2, delta3, self.v, self.p, self.dx_[n], w=self.alevar['w'], Fale=self.alevar['Fale'])
                    self.deltaW_int_old += self.vf.stab_v(delta1, delta2, delta3, self.v_old, self.p_old, self.dx_[n], w=self.alevar['w_old'], Fale=self.alevar['Fale_old'])

                    self.deltaW_p       += self.vf.stab_p(delta1, delta3, self.v, self.p, self.rho[n], self.dx_[n], w=self.alevar['w'], Fale=self.alevar['Fale'])
                    self.deltaW_p_old   += self.vf.stab_p(delta1, delta3, self.v_old, self.p_old, self.rho[n], self.dx_[n], w=self.alevar['w_old'], Fale=self.alevar['Fale_old'])

        ### full weakforms

        # kinetic plus internal minus external virtual power
        self.weakform_v = self.timefac_m * self.deltaW_kin + (1.-self.timefac_m) * self.deltaW_kin_old + \
                          self.timefac   * self.deltaW_int + (1.-self.timefac)   * self.deltaW_int_old - \
                          self.timefac   * self.deltaW_ext - (1.-self.timefac)   * self.deltaW_ext_old

        if self.pressure_at_midpoint:
            self.weakform_p = self.timefac   * self.deltaW_p   + (1.-self.timefac)   * self.deltaW_p_old
        else:
            self.weakform_p = self.deltaW_p

        self.weakform_lin_vv = ufl.derivative(self.weakform_v, self.v, self.dv)
        self.weakform_lin_vp = ufl.derivative(self.weakform_v, self.p, self.dp)
        self.weakform_lin_pv = ufl.derivative(self.weakform_p, self.v, self.dv)
        if self.stabilization is not None:
            self.weakform_lin_pp = ufl.derivative(self.weakform_p, self.p, self.dp)

        if self.prestress_initial:
            # prestressing weak forms
            self.weakform_prestress_v = self.deltaW_prestr_kin + self.deltaW_int - self.deltaW_prestr_ext
            self.weakform_lin_prestress_vv = ufl.derivative(self.weakform_prestress_v, self.v, self.dv)
            self.weakform_prestress_p = self.deltaW_p
            self.weakform_lin_prestress_vp = ufl.derivative(self.weakform_prestress_v, self.p, self.dp)
            self.weakform_lin_prestress_pv = ufl.derivative(self.weakform_prestress_p, self.v, self.dv)
            if self.stabilization is not None:
                self.weakform_lin_prestress_pp = ufl.derivative(self.weakform_prestress_p, self.p, self.dp)


    # active stress ODE evaluation - for reduced solid model
    def evaluate_active_stress_ode(self, t):

        tau_a_, na = [], 0
        for nm in range(len(self.bc_dict['membrane'])):

            if self.mem_active_stress[nm]:
                tau_a_.append(self.actstress[na].tau_act(self.tau_a_old, t, self.dt))
                na+=1
            else:
                tau_a_.append(ufl.as_ufl(0))

        # project and interpolate to quadrature function space
        tau_a_proj = project(tau_a_, self.Vd_scalar, self.dx_) # TODO: Should be self.dbmem here, but yields error; why?
        self.tau_a.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        self.tau_a.interpolate(tau_a_proj)


    # rate equations
    def evaluate_rate_equations(self, t_abs, t_off=0):

        # take care of active stress
        if self.have_active_stress and self.active_stress_trig == 'ode':
            self.evaluate_active_stress_ode(t_abs-t_off)


    def set_problem_residual_jacobian_forms(self):

        tes = time.time()
        if self.comm.rank == 0:
            print('FEM form compilation...')
            sys.stdout.flush()

        if not self.prestress_initial or self.restart_step > 0:
            self.res_v = fem.form(self.weakform_v)
            self.res_p = fem.form(self.weakform_p)
            self.jac_vv = fem.form(self.weakform_lin_vv)
            self.jac_vp = fem.form(self.weakform_lin_vp)
            self.jac_pv = fem.form(self.weakform_lin_pv)
            if self.stabilization is not None:
                self.jac_pp = fem.form(self.weakform_lin_pp)
        else:
            self.res_v  = fem.form(self.weakform_prestress_v)
            self.jac_vv = fem.form(self.weakform_lin_prestress_vv)
            self.res_p  = fem.form(self.weakform_prestress_p)
            self.jac_vp = fem.form(self.weakform_lin_prestress_vp)
            self.jac_pv = fem.form(self.weakform_lin_prestress_pv)
            if self.stabilization is not None:
                self.jac_pp = fem.form(self.weakform_lin_prestress_pp)

        tee = time.time() - tes
        if self.comm.rank == 0:
            print('FEM form compilation finished, te = %.2f s' % (tee))
            sys.stdout.flush()


    def assemble_residual_stiffness(self, t, subsolver=None):

        # assemble velocity rhs vector
        r_v = fem.petsc.assemble_vector(self.res_v)
        fem.apply_lifting(r_v, [self.jac_vv], [self.bc.dbcs], x0=[self.v.vector], scale=-1.0)
        r_v.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(r_v, self.bc.dbcs, x0=self.v.vector, scale=-1.0)

        # assemble system matrix
        K_vv = fem.petsc.assemble_matrix(self.jac_vv, self.bc.dbcs)
        K_vv.assemble()

        # assemble pressure rhs vector
        r_p = fem.petsc.assemble_vector(self.res_p)
        r_p.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # assemble system matrices
        K_vp = fem.petsc.assemble_matrix(self.jac_vp, self.bc.dbcs)
        K_vp.assemble()
        K_pv = fem.petsc.assemble_matrix(self.jac_pv, []) # currently, we do not consider pressure DBCs
        K_pv.assemble()
        if self.stabilization is not None:
            K_pp = fem.petsc.assemble_matrix(self.jac_pp, []) # currently, we do not consider pressure DBCs
        else:
            K_pp = None

        return [r_v, r_p], [[K_vv, K_vp], [K_pv, K_pp]]


    def get_index_sets(self, isoptions={}):

        if self.have_rom: # currently, ROM can only be on (subset of) first variable
            vred = PETSc.Vec().createMPI(size=(self.rom.V.getLocalSize()[1],self.rom.V.getSize()[1]), comm=self.comm)
            self.rom.V.multTranspose(self.v.vector, vred)
            vvec = vred
        else:
            vvec = self.v.vector

        offset_v = vvec.getOwnershipRange()[0] + self.p.vector.getOwnershipRange()[0]
        iset_v = PETSc.IS().createStride(vvec.getLocalSize(), first=offset_v, step=1, comm=self.comm)

        offset_p = offset_v + vvec.getLocalSize()
        iset_p = PETSc.IS().createStride(self.p.vector.getLocalSize(), first=offset_p, step=1, comm=self.comm)

        return [iset_v, iset_p]


    ### now the base routines for this problem

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
        self.ti.set_time_funcs(t, self.ti.funcs_to_update, self.ti.funcs_to_update_vec)

        # evaluate rate equations
        self.evaluate_rate_equations(t)


    def evaluate_post_solve(self, t, N):
        pass


    def set_output_state(self):
        pass


    def write_output(self, N, t, mesh=False):

        self.io.write_output(self, N=N, t=t)


    def update(self):

        # update - velocity, acceleration, pressure, all internal variables, all time functions
        self.ti.update_timestep(self.v, self.v_old, self.a_old, self.p, self.p_old, self.internalvars, self.internalvars_old, uf_old=self.uf_old)


    def print_to_screen(self):
        pass


    def induce_state_change(self):
        pass


    def write_restart(self, sname, N):

        self.io.write_restart(self, N)


    def check_abort(self, t):
        pass



class FluidmechanicsSolver(solver_base):

    def initialize_nonlinear_solver(self):

        self.pb.set_problem_residual_jacobian_forms()

        # perform Proper Orthogonal Decomposition
        if self.pb.have_rom:
            self.pb.rom.prepare_rob()

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear(self.pb, solver_params=self.solver_params)


    def solve_initial_state(self):

        # consider consistent initial acceleration
        if (self.pb.fluid_governing_type == 'navierstokes_transient' or self.pb.fluid_governing_type == 'stokes_transient') and self.pb.restart_step == 0:
            # weak form at initial state for consistent initial acceleration solve
            weakform_a = self.pb.deltaW_kin_old + self.pb.deltaW_int_old - self.pb.deltaW_ext_old

            weakform_lin_aa = ufl.derivative(weakform_a, self.pb.a_old, self.pb.dv) # actually linear in a_old

            # solve for consistent initial acceleration a_old
            self.solnln.solve_consistent_ini_acc(weakform_a, weakform_lin_aa, self.pb.a_old)


    def solve_nonlinear_problem(self, t):

        self.solnln.newton(t)


    def print_timestep_info(self, N, t, wt):

        # print time step info to screen
        self.pb.ti.print_timestep(N, t, self.solnln.sepstring, wt=wt)


    def solve_initial_prestress(self):

        utilities.print_prestress('start', self.pb.comm)

        if self.pb.prestress_ptc: self.solnln.PTC = True

        dt_prestr = self.pb.prestress_maxtime/self.pb.prestress_numstep

        for N in range(1,self.pb.prestress_numstep+1):

            wts = time.time()

            tprestr = N * dt_prestr

            self.pb.ti.set_time_funcs(tprestr, self.pb.funcs_to_update_pre, self.pb.funcs_to_update_vec_pre)

            self.solnln.newton(0.0)

            # MULF update - use backward scheme to calculate uf
            uf_vec = float(dt_prestr) * self.pb.v.vector + self.pb.uf_old.vector
            self.pb.ki.prestress_update(uf_vec)
            utilities.print_prestress('updt', self.pb.comm)

            wt = time.time() - wts

            # update fluid displacement: uf_old <- uf
            self.pb.uf_old.vector.axpby(1.0, 0.0, uf_vec)
            self.pb.uf_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            # update fluid velocity: v_old <- v
            self.pb.v_old.vector.axpby(1.0, 0.0, self.pb.v.vector)
            self.pb.v_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            # print time step info to screen
            self.pb.ti.print_prestress_step(N, tprestr, self.pb.prestress_numstep, self.solnln.sepstring, wt=wt)

            # write prestress displacement (given that we want to write the fluid displacement)
            if 'fluiddisplacement' in self.pb.results_to_write:
                self.pb.io.write_output_pre(self.pb, self.pb.uf_pre, tprestr, 'fluiddisplacement_pre')

        utilities.print_prestress('end', self.pb.comm)

        # reset state
        self.pb.uf_old.vector.set(0.0)
        self.pb.uf_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.pb.v.vector.set(0.0)
        self.pb.v.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.pb.v_old.vector.set(0.0)
        self.pb.v_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.pb.a_old.vector.set(0.0)
        self.pb.a_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # reset PTC flag to what it was
        if self.pb.prestress_ptc:
            try: self.solnln.PTC = self.solver_params['ptc']
            except: self.solnln.PTC = False

        # set flag to false again
        self.pb.prestress_initial = False
        self.pb.set_problem_residual_jacobian_forms()
