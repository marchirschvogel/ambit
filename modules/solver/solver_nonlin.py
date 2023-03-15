#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys, time

import numpy as np
from petsc4py import PETSc
from dolfinx import fem
import ufl

from projection import project
from mpiroutines import allgather_vec

from solver_utils import sol_utils
import preconditioner

### useful infos for PETSc mats, vecs, solvers...
# https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.Mat-class.html
# https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.Vec-class.html
# https://www.mcs.anl.gov/petsc/documentation/faq.html
# https://www.mcs.anl.gov/petsc/documentation/linearsolvertable.html
# https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.PC-class.html

# standard nonlinear solver for FEM problems
class solver_nonlinear:
    
    def __init__(self, pb, V_u, V_p=None, solver_params={}):

        self.pb = pb
        self.V_u = V_u
        self.V_p = V_p
        
        self.ptype = self.pb.problem_physics
        
        self.set_solver_params(solver_params)
        
        if self.pb.incompressible_2field:
            self.tolerances = {'res_u' : self.tolres, 'inc_u' : self.tolinc, 'res_p' : self.tolres, 'inc_p' : self.tolinc}
        else:
            self.tolerances = {'res_u' : self.tolres, 'inc_u' : self.tolinc}

        self.solutils = sol_utils(self.pb, self.ptype, solver_params)
        self.sepstring = self.solutils.timestep_separator(self.tolerances)
        
        self.initialize_petsc_solver()
        

    def set_solver_params(self, solver_params):
        
        try: self.maxiter = solver_params['maxiter']
        except: self.maxiter = 25

        try: self.divcont = solver_params['divergence_continue']
        except: self.divcont = None
        
        try: self.PTC = solver_params['ptc']
        except: self.PTC = False
        
        try: self.k_PTC_initial = solver_params['k_ptc_initial']
        except: self.k_PTC_initial = 0.1

        try: self.PTC_randadapt_range = solver_params['ptc_randadapt_range']
        except: self.PTC_randadapt_range = [0.85, 1.35]
        
        try: self.direct_solver = solver_params['direct_solver']
        except: self.direct_solver = 'superlu_dist'
        
        try: self.adapt_linsolv_tol = solver_params['adapt_linsolv_tol']
        except: self.adapt_linsolv_tol = False
        
        try: self.adapt_factor = solver_params['adapt_factor']
        except: self.adapt_factor = 0.1
        
        try: self.tollin = solver_params['tol_lin']
        except: self.tollin = 1.0e-8

        try: self.maxliniter = solver_params['max_liniter']
        except: self.maxliniter = 1200
        
        try: self.print_local_iter = solver_params['print_local_iter']
        except: self.print_local_iter = False
        
        try: self.tol_res_local = solver_params['tol_res_local']
        except: self.tol_res_local = 1.0e-10
        
        try: self.tol_inc_local = solver_params['tol_inc_local']
        except: self.tol_inc_local = 1.0e-10

        self.solvetype = solver_params['solve_type']
        self.tolres = solver_params['tol_res']
        self.tolinc = solver_params['tol_inc']


    def initialize_petsc_solver(self):
        
        # set forms to use (differ in case of initial prestress)
        self.pb.set_forms_solver()

        # create solver
        self.ksp = PETSc.KSP().create(self.pb.comm)
    
        # offset for pressure block
        if self.pb.incompressible_2field:
            self.Vu_map = self.V_u.dofmap.index_map
            self.offsetp = self.Vu_map.size_local * self.V_u.dofmap.index_map_bs
        
        if self.solvetype=='direct':
            
            self.ksp.setType("preonly")
            self.ksp.getPC().setType("lu")
            self.ksp.getPC().setFactorSolverType(self.direct_solver)
            
        elif self.solvetype=='iterative':
            
            # 2x2 block iterative method
            if self.pb.incompressible_2field:

                self.ksp.setType("gmres")
                self.ksp.getPC().setType("fieldsplit")
                # TODO: What is the difference btw. ADDITIVE, MULTIPLICATIVE, SCHUR, SPECIAL?
                self.ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
                #self.ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
                #self.ksp.getPC().setFieldSplitSchurFactType(0)

                # build "dummy" nested matrix in order to get the nested ISs (index sets)
                locmatsize_u, locmatsize_p = self.V_u.dofmap.index_map.size_local * self.V_u.dofmap.index_map_bs, self.V_p.dofmap.index_map.size_local * self.V_p.dofmap.index_map_bs
                matsize_u, matsize_p = self.V_u.dofmap.index_map.size_global * self.V_u.dofmap.index_map_bs, self.V_p.dofmap.index_map.size_global * self.V_p.dofmap.index_map_bs
                K_uu = PETSc.Mat().createAIJ(size=((locmatsize_u,matsize_u),(locmatsize_u,matsize_u)), bsize=None, nnz=None, csr=None, comm=self.pb.comm)
                K_uu.setUp()
                K_up = PETSc.Mat().createAIJ(size=((locmatsize_u,matsize_u),(locmatsize_p,matsize_p)), bsize=None, nnz=None, csr=None, comm=self.pb.comm)
                K_up.setUp()                
                K_pu = PETSc.Mat().createAIJ(size=((locmatsize_p,matsize_p),(locmatsize_u,matsize_u)), bsize=None, nnz=None, csr=None, comm=self.pb.comm)
                K_pu.setUp()  
                
                K_nest = PETSc.Mat().createNest([[K_uu, K_up], [K_pu, None]], isrows=None, iscols=None, comm=self.pb.comm)
                
                nested_IS = K_nest.getNestISs()
                self.ksp.getPC().setFieldSplitIS(
                    ("u", nested_IS[0][0]),
                    ("p", nested_IS[0][1]))

                # set the preconditioners for each block
                ksp_u, ksp_p = self.ksp.getPC().getFieldSplitSubKSP()
                
                # AMG for displacement/velocity block
                ksp_u.setType("preonly")
                ksp_u.getPC().setType("hypre")
                ksp_u.getPC().setMGLevels(3)
                ksp_u.getPC().setHYPREType("boomeramg")
                
                # AMG for pressure block
                ksp_p.setType("preonly")
                ksp_p.getPC().setType("hypre")
                ksp_p.getPC().setMGLevels(3)
                ksp_p.getPC().setHYPREType("boomeramg")

            else:
                
                # AMG
                self.ksp.getPC().setType("hypre")
                self.ksp.getPC().setMGLevels(3)
                self.ksp.getPC().setHYPREType("boomeramg")
            
            # set tolerances and print routine
            self.ksp.setTolerances(rtol=self.tollin, atol=None, divtol=None, max_it=self.maxliniter)
            self.ksp.setMonitor(lambda ksp, its, rnorm: self.solutils.print_linear_iter(its,rnorm))

        else:
            
            raise NameError("Unknown solvetype!")
        

    # solve for consistent initial acceleration a_old
    def solve_consistent_ini_acc(self, weakform_old, jac_a, a_old):

        # create solver
        ksp = PETSc.KSP().create(self.pb.comm)
        
        if self.solvetype=='direct':
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType(self.direct_solver)
        elif self.solvetype=='iterative':
            ksp.getPC().setType("hypre")
            ksp.getPC().setMGLevels(3)
            ksp.getPC().setHYPREType("boomeramg")
        else:
            raise NameError("Unknown solvetype!")
            
        # solve for consistent initial acceleration a_old
        M_a = fem.petsc.assemble_matrix(fem.form(jac_a), [])
        M_a.assemble()
        
        r_a = fem.petsc.assemble_vector(fem.form(weakform_old))
        r_a.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        ksp.setOperators(M_a)
        ksp.solve(-r_a, a_old.vector)
        
        a_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        ksp.destroy()
    

    def solve_local(self, localdata):
        
        for l in range(len(localdata['var'])):
            self.newton_local(localdata['var'][l],localdata['res'][l],localdata['inc'][l],localdata['fnc'][l])
        

    def newton(self, u, p=None, localdata={}):

        # displacement/velocity increment
        del_u_func = fem.Function(self.V_u)
        del_u = del_u_func.vector
        if self.pb.incompressible_2field:
            del_p_func = fem.Function(self.V_p)
            del_p = del_p_func.vector
        
        # get start vector in case we need to reset the nonlinear solver
        u_start = u.vector.duplicate()
        u_start.axpby(1.0, 0.0, u.vector)
        if self.pb.incompressible_2field:
            p_start = p.vector.duplicate()
            p_start.axpby(1.0, 0.0, p.vector)

        # Newton iteration index
        it = 0
        # for PTC
        k_PTC = self.k_PTC_initial
        counter_adapt, max_adapt = 0, 50
        maxresval = 1.0e16

        self.solutils.print_nonlinear_iter(header=True)

        while it < self.maxiter and counter_adapt < max_adapt:

            tes = time.time()
            
            if self.pb.localsolve:
                self.solve_local(localdata)

            r_u, K_uu = self.pb.assemble_residual_stiffness_main(u)
      
            if self.pb.incompressible_2field:
                r_p, K_pp, K_up, K_pu = self.pb.assemble_residual_stiffness_incompressible()

            if self.PTC:
                # computes K_uu + k_PTC * I
                K_uu.shift(k_PTC)

            # model order reduction stuff
            if self.pb.have_rom and not self.pb.prestress_initial:
                tmp = K_uu.matMult(self.pb.rom.V) # K_uu * V
                K_uu = self.pb.rom.V.transposeMatMult(tmp) # V^T * K_uu * V
                r_u_, del_u_ = self.pb.rom.V.createVecRight(), self.pb.rom.V.createVecRight()
                self.pb.rom.V.multTranspose(r_u, r_u_) # V^T * r_u
                # deal with penalties that may be added to reduced residual to penalize certain modes
                if bool(self.pb.rom.redbasisvec_penalties):
                    u_ = K_uu.createVecRight()
                    self.pb.rom.V.multTranspose(u.vector, u_) # V^T * u
                    penterm_ = self.pb.rom.V.createVecRight()
                    self.pb.rom.Cpen.mult(u_, penterm_) # Cpen * V^T * u
                    r_u_.axpy(1.0, penterm_) # add penalty term to reduced residual
                    K_uu.aypx(1.0, self.pb.rom.CpenVTV) # K_uu + Cpen * V^T * V
                r_u, del_u = r_u_, del_u_
                if self.pb.incompressible_2field:
                    # offdiagonal pressure blocks
                    offdg1 = self.pb.rom.V.transposeMatMult(K_up) # V^T * K_up
                    offdg2 = K_pu.matMult(self.pb.rom.V) # K_pu * V
                    K_up, K_pu = offdg1, offdg2
                    # new offset for pressure block
                    self.offsetp = self.pb.rom.V.getLocalSize()[1]

            te = time.time() - tes

            if self.pb.incompressible_2field:
                
                tes = time.time()

                # nested u-p vector
                r_2field_nest = PETSc.Vec().createNest([r_u, r_p])
                
                # nested uu-up,pu-zero matrix
                K_2field_nest = PETSc.Mat().createNest([[K_uu, K_up], [K_pu, K_pp]], isrows=None, iscols=None, comm=self.pb.comm)
                K_2field_nest.assemble()

                te += time.time() - tes
                
                # for monolithic direct solver
                if self.solvetype=='direct':
                    
                    tes = time.time()
                    
                    K_2field = PETSc.Mat()
                    K_2field_nest.convert("aij", out=K_2field)
            
                    K_2field.assemble()
                    
                    r_2field = PETSc.Vec().createWithArray(r_2field_nest.getArray())
                    r_2field.assemble()

                    del_2field = K_2field.createVecLeft()
                    self.ksp.setOperators(K_2field)
                    te += time.time() - tes
                    
                    tss = time.time()
                    self.ksp.solve(-r_2field, del_2field)
                    ts = time.time() - tss
                
                # for nested iterative solver
                elif self.solvetype=='iterative': 

                    tes = time.time()

                    P_pp = fem.petsc.assemble_matrix(fem.form(self.pb.a_p11), [])
                    P = PETSc.Mat().createNest([[K_uu, None], [None, P_pp]])
                    P.assemble()

                    del_2field = PETSc.Vec().createNest([del_u, del_p])
                    self.ksp.setOperators(K_2field_nest, P)
                    
                    te += time.time() - tes
                    
                    tss = time.time()
                    self.ksp.solve(-r_2field_nest, del_2field)
                    ts = time.time() - tss
                    
                    self.solutils.print_linear_iter_last(self.ksp.getIterationNumber(),self.ksp.getResidualNorm())
                    
                    if self.adapt_linsolv_tol:
                        self.solutils.adapt_linear_solver(r_u.norm())

                else:
                    
                    raise NameError("Unknown solvetype!")
                    
                del_u.array[:] = del_2field.array_r[:self.offsetp]
                del_p.array[:] = del_2field.array_r[self.offsetp:]
                
            else:
                    
                # solve linear system
                self.ksp.setOperators(K_uu)
                
                tss = time.time()
                self.ksp.solve(-r_u, del_u)
                ts = time.time() - tss
            
                if self.solvetype=='iterative':
                    
                    self.solutils.print_linear_iter_last(self.ksp.getIterationNumber(),self.ksp.getResidualNorm())
                        
                    if self.adapt_linsolv_tol:
                        self.solutils.adapt_linear_solver(r_u.norm())

            # get solid/fluid residual and increment norms
            resnorms = {'res_u' : r_u.norm()}
            incnorms = {'inc_u' : del_u.norm()}
            
            # reconstruct full-length increment vector
            if self.pb.have_rom and not self.pb.prestress_initial:
                del_u = self.pb.rom.V.createVecLeft()
                self.pb.rom.V.mult(del_u_, del_u) # V * d_red
            
            # update displacement/velocity solution
            u.vector.axpy(1.0, del_u)
            u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            if self.pb.incompressible_2field:
                # get pressure residual and increment norms
                resnorms['res_p'] = r_p.norm()
                incnorms['inc_p'] = del_p.norm()
                # update pressure solution
                p.vector.axpy(1.0, del_p)
                p.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            self.solutils.print_nonlinear_iter(it,resnorms,incnorms,self.PTC,k_PTC,ts=ts,te=te)
            
            it += 1
            
            # for PTC
            if self.PTC and it > 1 and struct_res_u_norm_last > 0.: k_PTC *= resnorms['res_u']/struct_res_u_norm_last
            struct_res_u_norm_last = resnorms['res_u']
            
            # adaptive PTC (for 3D block K_uu only!)
            if self.divcont=='PTC':
                
                self.maxiter = 250
                err = self.solutils.catch_solver_errors(resnorms['res_u'], incnorm=incnorms['inc_u'], maxval=maxresval)
                
                if err:
                    self.PTC = True
                    # reset Newton step
                    it, k_PTC = 0, self.k_PTC_initial
                    k_PTC *= np.random.uniform(self.PTC_randadapt_range[0], self.PTC_randadapt_range[1])
                    self.reset_step(u.vector,u_start,True)
                    if self.pb.incompressible_2field: self.reset_step(p.vector,p_start,True)
                    counter_adapt += 1
            
            # check if converged
            converged = self.solutils.check_converged(resnorms,incnorms,self.tolerances)
            if converged:
                if self.divcont=='PTC':
                    self.PTC = False
                    counter_adapt = 0
                break
        
        else:

            raise RuntimeError("Newton did not converge after %i iterations!" % (it))


    def reset_step(self, vec, vec_start, ghosted):
        
        vec.axpby(1.0, 0.0, vec_start)
        
        if ghosted:
            vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


    # local Newton where increment can be expressed as form at integration point level
    def newton_local(self, var, residual_forms, increment_forms, functionspaces, maxiter_local=20):

        it_local = 0
        
        num_loc_res = len(residual_forms)
        
        residuals, increments = [], []
        
        for i in range(num_loc_res):
            residuals.append(fem.Function(functionspaces[i]))
            increments.append(fem.Function(functionspaces[i]))

        res_norms, inc_norms = np.ones(num_loc_res), np.ones(num_loc_res)

        # return mapping scheme for nonlinear constitutive laws
        while it_local < maxiter_local:

            for i in range(num_loc_res):
                
                # interpolate symbolic increment form into increment vector
                increment_proj = project(increment_forms[i], functionspaces[i], self.pb.dx_)
                increments[i].vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                increments[i].interpolate(increment_proj)
                
            for i in range(num_loc_res):
                # update var vector
                var[i].vector.axpy(1.0, increments[i].vector)
                var[i].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            for i in range(num_loc_res):
                # interpolate symbolic residual form into residual vector
                residual_proj = project(residual_forms[i], functionspaces[i], self.pb.dx_)
                residuals[i].vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                residuals[i].interpolate(residual_proj)
                # get residual and increment inf norms
                res_norms[i] = residuals[i].vector.norm(norm_type=3)
                inc_norms[i] = increments[i].vector.norm(norm_type=3)
            
            if self.print_local_iter:
                if self.pb.comm.rank == 0:
                    print("      (it_local = %i, res: %.4e, inc: %.4e)" % (it_local,np.sum(res_norms),np.sum(inc_norms)))
                    sys.stdout.flush()
            
            # increase iteration index
            it_local += 1
            
            # check if converged
            if np.sum(res_norms) <= self.tol_res_local and np.sum(inc_norms) <= self.tol_inc_local:

                break
            
        else:

            raise RuntimeError("Local Newton did not converge after %i iterations!" % (it_local))



# nonlinear solver for Lagrange multiplier constraints and 3D-0D coupled monolithic formulations
class solver_nonlinear_constraint_monolithic(solver_nonlinear):
    
    def __init__(self, pb, V_u, V_p=None, solver_params_3D={}, solver_params_constr={}):
        
        # coupled problem
        self.pb = pb
        self.V_u = V_u
        self.V_p = V_p
        
        self.ptype = self.pb.problem_physics

        self.set_solver_params(solver_params_3D)

        self.solver_params_3D = solver_params_3D
        self.solver_params_constr = solver_params_constr

        self.ptype = self.pb.problem_physics
        self.solutils = sol_utils(self.pb, self.ptype, solver_params_3D)

        self.tolres0D = solver_params_constr['tol_res']
        self.tolinc0D = solver_params_constr['tol_inc']

        if self.pb.pbs.incompressible_2field:
            self.tolerances = {'res_u' : self.tolres, 'inc_u' : self.tolinc, 'res_p' : self.tolres, 'inc_p' : self.tolinc, 'res_0d' : self.tolres0D, 'inc_0d' : self.tolinc0D}
            # dof offset for pressure block and 0D model
            self.V3D_map_u = V_u.dofmap.index_map
            self.V3D_map_p = V_p.dofmap.index_map
            self.offsetp = self.V3D_map_u.size_local * V_u.dofmap.index_map_bs
            self.offset0D = self.V3D_map_u.size_local * V_u.dofmap.index_map_bs + self.V3D_map_p.size_local * V_p.dofmap.index_map_bs
        else:
            self.tolerances = {'res_u' : self.tolres, 'inc_u' : self.tolinc, 'res_0d' : self.tolres0D, 'inc_0d' : self.tolinc0D}
            # dof offset for 0D model
            self.V3D_map_u = V_u.dofmap.index_map
            self.offset0D = self.V3D_map_u.size_local * V_u.dofmap.index_map_bs

        # initialize 0D solver class for monolithic Lagrange multiplier coupling
        if self.pb.coupling_type == 'monolithic_lagrange' and (self.ptype == 'solid_flow0d' or self.ptype == 'fluid_flow0d'):
            self.snln0D = solver_nonlinear_ode(self.pb.pbf, self.solver_params_constr)

        self.sepstring = self.solutils.timestep_separator(self.tolerances)
        
        self.initialize_petsc_solver()
        
        
    def initialize_petsc_solver(self):

        # set forms to use (differ in case of initial prestress)
        self.pb.pbs.set_forms_solver()

        # create solver
        self.ksp = PETSc.KSP().create(self.pb.comm)
        
        # 0D flow, or Lagrange multiplier system matrix
        if self.pb.coupling_type == 'monolithic_direct': self.K_ss = self.pb.pbf.K
        if self.pb.coupling_type == 'monolithic_lagrange': self.K_ss = self.pb.K_lm
        
        if self.solvetype=='direct':
            
            self.ksp.setType("preonly")
            self.ksp.getPC().setType("lu")
            self.ksp.getPC().setFactorSolverType(self.direct_solver)
        
        elif self.solvetype=='iterative':

            # 3x3 block iterative method
            if self.pb.pbs.incompressible_2field:

                self.ksp.setType("gmres")
                self.ksp.getPC().setType("fieldsplit")
                # TODO: What is the difference btw. ADDITIVE, MULTIPLICATIVE, SCHUR, SPECIAL?
                self.ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
                #self.ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
                #self.ksp.getPC().setFieldSplitSchurFactType(0)

                # build "dummy" nested matrix in order to get the nested ISs (index sets)
                locmatsize_u, locmatsize_p = self.V_u.dofmap.index_map.size_local * self.V_u.dofmap.index_map_bs, self.V_p.dofmap.index_map.size_local * self.V_p.dofmap.index_map_bs
                matsize_u, matsize_p = self.V_u.dofmap.index_map.size_global * self.V_u.dofmap.index_map_bs, self.V_p.dofmap.index_map.size_global * self.V_p.dofmap.index_map_bs
                K_uu = PETSc.Mat().createAIJ(size=((locmatsize_u,matsize_u),(locmatsize_u,matsize_u)), bsize=None, nnz=None, csr=None, comm=self.pb.comm)
                K_uu.setUp()
                K_up = PETSc.Mat().createAIJ(size=((locmatsize_u,matsize_u),(locmatsize_p,matsize_p)), bsize=None, nnz=None, csr=None, comm=self.pb.comm)
                K_up.setUp()                
                K_pu = PETSc.Mat().createAIJ(size=((locmatsize_p,matsize_p),(locmatsize_u,matsize_u)), bsize=None, nnz=None, csr=None, comm=self.pb.comm)
                K_pu.setUp()                   
                
                K_nest = PETSc.Mat().createNest([[K_uu, K_up, None], [K_pu, None, None], [None, None, self.K_ss]], isrows=None, iscols=None, comm=self.pb.comm)
                
                nested_IS = K_nest.getNestISs()
                self.ksp.getPC().setFieldSplitIS(
                    ("u", nested_IS[0][0]),
                    ("p", nested_IS[0][1]),
                    ("s", nested_IS[0][2]))

                # set the preconditioners for each block
                ksp_u, ksp_p, ksp_s = self.ksp.getPC().getFieldSplitSubKSP()
                
                # AMG for displacement/velocity block
                ksp_u.setType("preonly")
                ksp_u.getPC().setType("hypre")
                ksp_u.getPC().setMGLevels(3)
                ksp_u.getPC().setHYPREType("boomeramg")
                
                # AMG for pressure block
                ksp_p.setType("preonly")
                ksp_p.getPC().setType("hypre")
                ksp_p.getPC().setMGLevels(3)
                ksp_p.getPC().setHYPREType("boomeramg")
                
                # direct solve for 0D block
                ksp_s.setType("preonly")
                ksp_s.getPC().setType("lu")
                
            # 2x2 block iterative method
            else:
                
                self.ksp.setType("gmres")
                self.ksp.getPC().setType("fieldsplit")
                # TODO: What is the difference btw. ADDITIVE, MULTIPLICATIVE, SCHUR, SPECIAL?
                self.ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
                #self.ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
                #self.ksp.getPC().setFieldSplitSchurFactType(0)
                
                # build "dummy" nested matrix in order to get the nested ISs (index sets)
                locmatsize = self.V_u.dofmap.index_map.size_local * self.V_u.dofmap.index_map_bs
                matsize = self.V_u.dofmap.index_map.size_global * self.V_u.dofmap.index_map_bs
                K_uu = PETSc.Mat().createAIJ(size=((locmatsize,matsize),(locmatsize,matsize)), bsize=None, nnz=None, csr=None, comm=self.pb.comm)
                K_uu.setUp()
                
                K_nest = PETSc.Mat().createNest([[K_uu, None], [None, self.K_ss]], isrows=None, iscols=None, comm=self.pb.comm)
                
                nested_IS = K_nest.getNestISs()
                self.ksp.getPC().setFieldSplitIS(
                    ("u", nested_IS[0][0]),
                    ("s", nested_IS[0][1]))

                # set the preconditioners for each block
                ksp_u, ksp_s = self.ksp.getPC().getFieldSplitSubKSP()
                
                # AMG for displacement/velocity block
                ksp_u.setType("preonly")
                ksp_u.getPC().setType("hypre")
                ksp_u.getPC().setMGLevels(3)
                ksp_u.getPC().setHYPREType("boomeramg")
                
                # direct solve for 0D block
                ksp_s.setType("preonly")
                ksp_s.getPC().setType("lu")

            # set tolerances and print routine
            self.ksp.setTolerances(rtol=self.tollin, atol=None, divtol=None, max_it=self.maxliniter)
            self.ksp.setMonitor(lambda ksp, its, rnorm: self.solutils.print_linear_iter(its,rnorm))

        else:
            
            raise NameError("Unknown solvetype!")


    def newton(self, u, p, s, t, localdata={}):
        
        # 3D displacement/velocity increment
        del_u_func = fem.Function(self.V_u)
        del_u = del_u_func.vector
        if self.pb.pbs.incompressible_2field:
            # 3D pressure increment
            del_p_func = fem.Function(self.V_p)
            del_p = del_p_func.vector
        # 0D/Lagrange multiplier increment
        del_s = self.K_ss.createVecLeft()
        
        # get start vectors in case we need to reset the nonlinear solver
        u_start = u.vector.duplicate()
        s_start = s.duplicate()
        u_start.axpby(1.0, 0.0, u.vector)
        s.assemble(), s_start.axpby(1.0, 0.0, s)
        if self.pb.pbs.incompressible_2field:
            p_start = p.vector.duplicate()
            p_start.axpby(1.0, 0.0, p.vector)

        # Newton iteration index
        it = 0
        # for PTC
        k_PTC = self.k_PTC_initial
        counter_adapt, max_adapt = 0, 50
        maxresval = 1.0e16
        
        self.solutils.print_nonlinear_iter(header=True)

        while it < self.maxiter and counter_adapt < max_adapt:
            
            tes = time.time()
            
            if self.ptype == 'solid_constraint':
                self.pb.lm.assemble()
                ls, le = self.pb.lm.getOwnershipRange()
            
            if self.pb.coupling_type == 'monolithic_lagrange' and (self.ptype == 'solid_flow0d' or self.ptype == 'fluid_flow0d'):
                self.pb.lm.assemble()
                ls, le = self.pb.lm.getOwnershipRange()
                # Lagrange multipliers (pressures) to be passed to 0D model
                lm_sq = allgather_vec(self.pb.lm, self.pb.comm)
                self.pb.pbf.c[:] = lm_sq[:]
                self.snln0D.newton(s, t, print_iter=self.pb.print_subiter, sub=True)

            if self.pb.pbs.localsolve:
                self.solve_local(localdata)

            # set the pressure functions for the load onto the 3D solid/fluid problem
            if self.pb.coupling_type == 'monolithic_direct':
                self.pb.pbf.cardvasc0D.set_pressure_fem(s, self.pb.pbf.cardvasc0D.v_ids, self.pb.pr0D, self.pb.coupfuncs)
            if self.pb.coupling_type == 'monolithic_lagrange' and (self.ptype == 'solid_flow0d' or self.ptype == 'fluid_flow0d'):
                self.pb.pbf.cardvasc0D.set_pressure_fem(self.pb.lm, list(range(self.pb.num_coupling_surf)), self.pb.pr0D, self.pb.coupfuncs)
            if self.pb.coupling_type == 'monolithic_lagrange' and self.ptype == 'solid_constraint':
                self.pb.set_pressure_fem(self.pb.lm, self.pb.coupfuncs)
            
            r_u, K_uu = self.pb.pbs.assemble_residual_stiffness_main(u)

            if self.PTC:
                # computes K_uu + k_PTC * I
                K_uu.shift(k_PTC)

            if self.pb.pbs.incompressible_2field:
                r_p, K_pp, K_up, K_pu = self.pb.pbs.assemble_residual_stiffness_incompressible()

            if self.pb.coupling_type == 'monolithic_direct':

                # volumes/fluxes to be passed to 0D model
                for i in range(len(self.pb.pbf.cardvasc0D.c_ids)):
                    cq = fem.assemble_scalar(fem.form(self.pb.cq[i]))
                    cq = self.pb.comm.allgather(cq)
                    self.pb.pbf.c[i] = sum(cq)*self.pb.cq_factor[i]

                # evaluate 0D model with current p and return df, f, K_ss
                self.pb.pbf.cardvasc0D.evaluate(s, t, self.pb.pbf.df, self.pb.pbf.f, self.pb.pbf.dK, self.pb.pbf.K, self.pb.pbf.c, self.pb.pbf.y, self.pb.pbf.aux)

                # 0D rhs vector and stiffness
                r_s, self.K_ss = self.pb.pbf.assemble_residual_stiffness(t)

            if self.ptype == 'solid_flow0d' or self.ptype == 'fluid_flow0d':
                # assemble 0D rhs contributions
                self.pb.pbf.df_old.assemble()
                self.pb.pbf.f_old.assemble()
                self.pb.pbf.df.assemble()
                self.pb.pbf.f.assemble()

            if self.pb.coupling_type == 'monolithic_lagrange' and (self.ptype == 'solid_flow0d' or self.ptype == 'fluid_flow0d'):

                for i in range(self.pb.num_coupling_surf):
                    cq = fem.assemble_scalar(fem.form(self.pb.cq[i]))
                    cq = self.pb.comm.allgather(cq)
                    self.pb.constr[i] = sum(cq)*self.pb.cq_factor[i]
                
                s_sq = allgather_vec(s, self.pb.comm)
                
                s_pert = self.pb.pbf.K.createVecLeft()
                s_pert.axpby(1.0, 0.0, s)

                # store df, f, and aux vectors prior to perturbation solves
                df_tmp, f_tmp, aux_tmp = self.pb.pbf.K.createVecLeft(), self.pb.pbf.K.createVecLeft(), np.zeros(self.pb.pbf.numdof)
                df_tmp.axpby(1.0, 0.0, self.pb.pbf.df)
                f_tmp.axpby(1.0, 0.0, self.pb.pbf.f)
                aux_tmp[:] = self.pb.pbf.aux[:]

                # finite differencing for LM siffness matrix
                for i in range(self.pb.num_coupling_surf):
                    for j in range(self.pb.num_coupling_surf):

                        self.pb.pbf.c[j] = lm_sq[j] + self.pb.eps_fd # perturbed LM
                        self.snln0D.newton(s_pert, t, print_iter=False)
                        s_pert_sq = allgather_vec(s_pert, self.pb.comm)
                        self.K_ss[i,j] = -(s_pert_sq[self.pb.pbf.cardvasc0D.v_ids[i]] - s_sq[self.pb.pbf.cardvasc0D.v_ids[i]])/self.pb.eps_fd
                        self.pb.pbf.c[j] = lm_sq[j] # restore LM

                # restore df, f, and aux vectors for correct time step update
                self.pb.pbf.df.axpby(1.0, 0.0, df_tmp)
                self.pb.pbf.f.axpby(1.0, 0.0, f_tmp)
                self.pb.pbf.aux[:] = aux_tmp[:]

            if self.ptype == 'solid_constraint':
                for i in range(len(self.pb.surface_p_ids)):
                    cq = fem.assemble_scalar(fem.form(self.pb.cq[i]))
                    cq = self.pb.comm.allgather(cq)
                    self.pb.constr[i] = sum(cq)*self.pb.cq_factor[i]

            if self.pb.coupling_type == 'monolithic_direct':
            
                # if we have prescribed variable values over time
                if bool(self.pb.pbf.prescribed_variables):
                    for a in self.pb.pbf.prescribed_variables:
                        varindex = self.pb.pbf.cardvasc0D.varmap[a]
                        curvenumber = self.pb.pbf.prescribed_variables[a]
                        val = self.pb.pbs.ti.timecurves(curvenumber)(t)
                        self.pb.pbf.cardvasc0D.set_prescribed_variables(s, r_s, self.K_ss, val, varindex)

            if self.pb.coupling_type == 'monolithic_lagrange' and (self.ptype == 'solid_flow0d' or self.ptype == 'fluid_flow0d'):

                r_s = self.K_ss.createVecLeft()
                
                s_old_sq = allgather_vec(self.pb.pbf.s_old, self.pb.comm)

                # Lagrange multiplier coupling residual
                for i in range(ls,le):
                    r_s[i] = self.pb.constr[i] - s_sq[self.pb.pbf.cardvasc0D.v_ids[i]]
                
            if self.pb.coupling_type == 'monolithic_lagrange' and self.ptype == 'solid_constraint':

                r_s = self.K_ss.createVecLeft()

                val, val_old = [], []
                for n in range(self.pb.num_coupling_surf):
                    curvenumber = self.pb.prescribed_curve[n]
                    val.append(self.pb.pbs.ti.timecurves(curvenumber)(t)), val_old.append(self.pb.pbs.ti.timecurves(curvenumber)(t-self.pb.pbs.dt))
    
                # Lagrange multiplier coupling residual
                for i in range(ls,le):
                    r_s[i] = self.pb.constr[i] - val[i]

            # 0D / Lagrange multiplier system matrix
            self.K_ss.assemble()

            if self.ptype == 'solid_flow0d' or self.ptype == 'fluid_flow0d': 
                if self.pb.coupling_type == 'monolithic_direct':
                    row_ids = self.pb.pbf.cardvasc0D.c_ids
                    col_ids = self.pb.pbf.cardvasc0D.v_ids
                if self.pb.coupling_type == 'monolithic_lagrange':
                    row_ids = list(range(self.pb.num_coupling_surf))
                    col_ids = list(range(self.pb.num_coupling_surf))

            if self.ptype == 'solid_constraint':    
                row_ids = list(range(self.pb.num_coupling_surf))
                col_ids = list(range(self.pb.num_coupling_surf))

            # offdiagonal u-s columns
            k_us_cols=[]
            for i in range(len(col_ids)):
                k_us_cols.append(fem.petsc.assemble_vector(fem.form(self.pb.dforce[i]))) # already multiplied by time-integration factor
        
            # offdiagonal s-u rows
            k_su_rows=[]
            for i in range(len(row_ids)):
                
                if self.ptype == 'solid_flow0d' or self.ptype == 'fluid_flow0d':
                    # depending on if we have volumes, fluxes, or pressures passed in (latter for LM coupling)
                    if self.pb.pbf.cq[i] == 'volume':   timefac = 1./self.pb.pbs.dt
                    if self.pb.pbf.cq[i] == 'flux':     timefac = -self.pb.pbf.theta0d_timint(t) # 0D model time-integration factor
                    if self.pb.pbf.cq[i] == 'pressure': timefac = 1.

                if self.ptype == 'solid_constraint': timefac = 1.
                
                k_su_rows.append(fem.petsc.assemble_vector(fem.form((timefac*self.pb.cq_factor[i])*self.pb.dcq[i])))

            # apply dbcs to matrix entries - basically since these are offdiagonal we want a zero there!
            for i in range(len(col_ids)):
                
                fem.apply_lifting(k_us_cols[i], [fem.form(self.pb.pbs.jac_uu)], [self.pb.pbs.bc.dbcs], x0=[u.vector], scale=0.0)
                k_us_cols[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                fem.set_bc(k_us_cols[i], self.pb.pbs.bc.dbcs, x0=u.vector, scale=0.0)
            
            for i in range(len(row_ids)):
            
                fem.apply_lifting(k_su_rows[i], [fem.form(self.pb.pbs.jac_uu)], [self.pb.pbs.bc.dbcs], x0=[u.vector], scale=0.0)
                k_su_rows[i].ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                fem.set_bc(k_su_rows[i], self.pb.pbs.bc.dbcs, x0=u.vector, scale=0.0)
            
            # setup offdiagonal matrices
            locmatsize = self.V3D_map_u.size_local * self.V_u.dofmap.index_map_bs
            matsize = self.V3D_map_u.size_global * self.V_u.dofmap.index_map_bs
            # row ownership range of uu block
            irs, ire = K_uu.getOwnershipRange()

            # derivative of solid/fluid residual w.r.t. 0D pressures
            K_us = PETSc.Mat().createAIJ(size=((locmatsize,matsize),(self.K_ss.getSize()[0])), bsize=None, nnz=None, csr=None, comm=self.pb.comm)
            K_us.setUp()

            # set columns
            for i in range(len(col_ids)):
                K_us[irs:ire, col_ids[i]] = k_us_cols[i][irs:ire]
                
            K_us.assemble()

            # derivative of 0D residual w.r.t. solid displacements/fluid velocities
            K_su = PETSc.Mat().createAIJ(size=((self.K_ss.getSize()[0]),(locmatsize,matsize)), bsize=None, nnz=None, csr=None, comm=self.pb.comm)
            K_su.setUp()

            # set rows
            for i in range(len(row_ids)):
                K_su[row_ids[i], irs:ire] = k_su_rows[i][irs:ire]

            K_su.assemble()

            # model order reduction stuff
            if self.pb.pbs.have_rom:
                tmp = K_uu.matMult(self.pb.pbs.rom.V) # K_uu * V
                K_uu = self.pb.pbs.rom.V.transposeMatMult(tmp) # V^T * K_uu * V
                r_u_, del_u_ = self.pb.pbs.rom.V.createVecRight(), self.pb.pbs.rom.V.createVecRight()
                self.pb.pbs.rom.V.multTranspose(r_u, r_u_) # V^T * r_u
                # deal with penalties that may be added to reduced residual to penalize certain modes
                if bool(self.pb.pbs.rom.redbasisvec_penalties):
                    u_ = K_uu.createVecRight()
                    self.pb.pbs.rom.V.multTranspose(u.vector, u_) # V^T * u
                    penterm_ = self.pb.pbs.rom.V.createVecRight()
                    self.pb.pbs.rom.Cpen.mult(u_, penterm_) # Cpen * V^T * u
                    r_u_.axpy(1.0, penterm_) # add penalty term to reduced residual
                    K_uu.aypx(1.0, self.pb.pbs.rom.CpenVTV) # K_uu + Cpen * V^T * V
                r_u, del_u = r_u_, del_u_
                # offdiagonal blocks
                offdg1 = self.pb.pbs.rom.V.transposeMatMult(K_us) # V^T * K_us
                offdg2 = K_su.matMult(self.pb.pbs.rom.V) # K_su * V
                K_us, K_su = offdg1, offdg2
                # set adequate offset for 0D/LM block
                self.offset0D = self.pb.pbs.rom.V.getLocalSize()[1]
                if self.pb.pbs.incompressible_2field:
                    # offdiagonal pressure blocks
                    offdg1 = self.pb.pbs.rom.V.transposeMatMult(K_up) # V^T * K_up
                    offdg2 = K_pu.matMult(self.pb.pbs.rom.V) # K_pu * V
                    K_up, K_pu = offdg1, offdg2
                    # new offset for pressure and 0D/LM block
                    self.offsetp = self.pb.pbs.rom.V.getLocalSize()[1]
                    self.offset0D = self.offsetp + self.V_p.dofmap.index_map.size_local * self.V_p.dofmap.index_map_bs

            if self.pb.pbs.incompressible_2field:
                K_3D0D_nest = PETSc.Mat().createNest([[K_uu, K_up, K_us], [K_pu, K_pp, None], [K_su, None, self.K_ss]], isrows=None, iscols=None, comm=self.pb.comm)
            else:
                K_3D0D_nest = PETSc.Mat().createNest([[K_uu, K_us], [K_su, self.K_ss]], isrows=None, iscols=None, comm=self.pb.comm)

            K_3D0D_nest.assemble()
            
            # 0D rhs vector
            r_s.assemble()
            
            # nested 3D-0D vector
            if self.pb.pbs.incompressible_2field:
                r_3D0D_nest = PETSc.Vec().createNest([r_u, r_p, r_s])
            else:
                r_3D0D_nest = PETSc.Vec().createNest([r_u, r_s])
            
            te = time.time() - tes
            
            # solve linear system
            
            # for monolithic direct solver
            if self.solvetype=='direct':
                
                tes = time.time()
                
                K_3D0D = PETSc.Mat()
                K_3D0D_nest.convert("aij", out=K_3D0D)
            
                K_3D0D.assemble()

                r_3D0D = PETSc.Vec().createWithArray(r_3D0D_nest.getArray())
                r_3D0D.assemble()

                del_sol = K_3D0D.createVecLeft()
                self.ksp.setOperators(K_3D0D)
                
                te += time.time() - tes
                
                tss = time.time()
                self.ksp.solve(-r_3D0D, del_sol)
                ts = time.time() - tss
                
            # for nested iterative solver
            elif self.solvetype=='iterative':
                
                tes = time.time()
                
                if self.pb.pbs.incompressible_2field:

                    # SIMPLE/block diagonal preconditioner
                    P_us = preconditioner.simple2x2(K_uu,K_us,K_su,self.K_ss)
                    P_pp = fem.petsc.assemble_matrix(fem.form(self.pb.pbs.a_p11), [])
                    P = PETSc.Mat().createNest([[P_us.getNestSubMatrix(0,0), None, P_us.getNestSubMatrix(0,1)], [P_us.getNestSubMatrix(1,0), P_pp, None], [None, None, P_us.getNestSubMatrix(1,1)]], isrows=None, iscols=None, comm=self.pb.comm)
                    P.assemble()
                    
                    ## block diagonal preconditioner
                    #P_pp = fem.petsc.assemble_matrix(fem.form(self.pb.pbs.a_p11), [])
                    #P = PETSc.Mat().createNest([[K_uu, None, None], [None, P_pp, None], [None, None, self.K_ss]], isrows=None, iscols=None, comm=self.pb.comm)
                    #P.assemble()
                    
                    del_sol = PETSc.Vec().createNest([del_u, del_p, del_s])
                    self.ksp.setOperators(K_3D0D_nest, P)
                
                else:
                    
                    # SIMPLE preconditioner
                    P = preconditioner.simple2x2(K_uu,K_us,K_su,self.K_ss)
                    
                    ## block diagonal preconditioner
                    #P = PETSc.Mat().createNest([[K_uu, None], [None, self.K_ss]], isrows=None, iscols=None, comm=self.pb.comm)
                    #P.assemble()
                    
                    del_sol = PETSc.Vec().createNest([del_u, del_s])
                    self.ksp.setOperators(K_3D0D_nest, P)
                
                te += time.time() - tes
                
                tss = time.time()
                self.ksp.solve(-r_3D0D_nest, del_sol)
                ts = time.time() - tss

                self.solutils.print_linear_iter_last(self.ksp.getIterationNumber(),self.ksp.getResidualNorm())
                
                if self.adapt_linsolv_tol:
                    self.solutils.adapt_linear_solver(r_u.norm())

            else:
                
                raise NameError("Unknown solvetype!")

            if self.pb.pbs.incompressible_2field:
                del_u.array[:] = del_sol.array_r[:self.offsetp]
                del_p.array[:] = del_sol.array_r[self.offsetp:self.offset0D]
                del_s.array[:] = del_sol.array_r[self.offset0D:]
            else:
                del_u.array[:] = del_sol.array_r[:self.offset0D]
                del_s.array[:] = del_sol.array_r[self.offset0D:]

            # get solid/fluid residual and increment norms
            resnorms = {'res_u' : r_u.norm()}
            incnorms = {'inc_u' : del_u.norm()}

            # reconstruct full-length increment vector
            if self.pb.pbs.have_rom:
                del_u = self.pb.pbs.rom.V.createVecLeft()
                self.pb.pbs.rom.V.mult(del_u_, del_u) # V * d_red

            if self.pb.pbs.incompressible_2field:
                # get pressure residual and increment norms
                resnorms['res_p'] = r_p.norm()
                incnorms['inc_p'] = del_p.norm()

            # get flow0d residual and increment norms
            resnorms['res_0d'] = r_s.norm()
            incnorms['inc_0d'] = del_s.norm()

            # update solution - displacement/velocity
            u.vector.axpy(1.0, del_u)
            u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            # update solution - pressure
            if self.pb.pbs.incompressible_2field:
                p.vector.axpy(1.0, del_p)
                p.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            # update solution - 0D variables (not ghosted!)
            if self.pb.coupling_type == 'monolithic_direct': s.axpy(1.0, del_s)
            # update solution - Lagrange multipliers (not ghosted!)
            if self.pb.coupling_type == 'monolithic_lagrange': self.pb.lm.axpy(1.0, del_s)

            self.solutils.print_nonlinear_iter(it,resnorms,incnorms,self.PTC,k_PTC,ts=ts,te=te)
            
            it += 1
            
            # for PTC
            if self.PTC and it > 1 and struct_res_u_norm_last > 0.: k_PTC *= resnorms['res_u']/struct_res_u_norm_last
            struct_res_u_norm_last = resnorms['res_u']
            
            # adaptive PTC (for 3D block K_uu only!)
            if self.divcont=='PTC':
                
                self.maxiter = 250
                err = self.solutils.catch_solver_errors(resnorms['res_u'], incnorm=incnorms['inc_u'], maxval=maxresval)
                
                if err:
                    self.PTC = True
                    # reset Newton step
                    it, k_PTC = 0, self.k_PTC_initial
                    k_PTC *= np.random.uniform(self.PTC_randadapt_range[0], self.PTC_randadapt_range[1])
                    self.reset_step(u.vector,u_start,True), self.reset_step(s,s_start,False)
                    if self.pb.pbs.incompressible_2field: self.reset_step(p.vector,p_start,True)
                    counter_adapt += 1
            
            # check if converged
            converged = self.solutils.check_converged(resnorms,incnorms,self.tolerances)
            if converged:
                if self.divcont=='PTC':
                    self.PTC = False
                    counter_adapt = 0
                break

        else:

            raise RuntimeError("Monolithic 3D-0D Newton did not converge after %i iterations!" % (it))


# solver for pure ODE (0D) problems (e.g. a system of first order ODEs integrated with One-Step-Theta method)
class solver_nonlinear_ode(solver_nonlinear):

    def __init__(self, pb, solver_params={}):

        self.pb = pb
        
        self.ptype = self.pb.problem_physics

        try: self.maxiter = solver_params['maxiter']
        except: self.maxiter = 25
        
        try: self.direct_solver = solver_params['direct_solver']
        except: self.direct_solver = 'superlu_dist'        

        self.tolres = solver_params['tol_res']
        self.tolinc = solver_params['tol_inc']

        self.tolerances = {'res_0d' : self.tolres, 'inc_0d' : self.tolinc}
        
        self.PTC = False # don't think we'll ever need PTC for the 0D ODE problem...

        self.solutils = sol_utils(self.pb, self.ptype, solver_params)
        
        self.sepstring = self.solutils.timestep_separator(self.tolerances)
        
        self.initialize_petsc_solver()
        
        
    def initialize_petsc_solver(self):
        
        # create solver
        self.ksp = PETSc.KSP().create(self.pb.comm)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")
        self.ksp.getPC().setFactorSolverType(self.direct_solver)


    def newton(self, s, t, print_iter=True, sub=False):

        # Newton iteration index
        it = 0
        
        if print_iter: self.solutils.print_nonlinear_iter(header=True,sub=sub)
        
        while it < self.maxiter:
            
            tes = time.time()

            self.pb.odemodel.evaluate(s, t, self.pb.df, self.pb.f, self.pb.dK, self.pb.K, self.pb.c, self.pb.y, self.pb.aux)
            
            # ODE rhs vector and stiffness matrix
            r, K = self.pb.assemble_residual_stiffness(t)

            # if we have prescribed variable values over time
            if bool(self.pb.prescribed_variables):
                for a in self.pb.prescribed_variables:
                    varindex = self.pb.odemodel.varmap[a]
                    curvenumber = self.pb.prescribed_variables[a]
                    val = self.pb.ti.timecurves(curvenumber)(t)
                    self.pb.odemodel.set_prescribed_variables(s, r, K, val, varindex)
            
            ds = K.createVecLeft()
            
            # solve linear system
            self.ksp.setOperators(K)
            
            te = time.time() - tes
            
            tss = time.time()
            self.ksp.solve(-r, ds)
            ts = time.time() - tss
            
            # update solution
            s.axpy(1.0, ds)
            
            # get norms
            res_norm = r.norm()
            inc_norm = ds.norm()
            
            if print_iter: self.solutils.print_nonlinear_iter(it,{'res_0d' : res_norm},{'inc_0d' : inc_norm},ts=ts,te=te,sub=sub)
            
            it += 1

            # check if converged
            converged = self.solutils.check_converged({'res_0d' : res_norm},{'inc_0d' : inc_norm},self.tolerances,ptype='flow0d')
            if converged:
                if print_iter and sub:
                    if self.pb.comm.rank == 0:
                        print('      **************************************************************')
                        print(' ')
                        sys.stdout.flush()
                break

        else:

            raise RuntimeError("Newton for ODE system did not converge after %i iterations!" % (it))
