#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys, time

import numpy as np
from petsc4py import PETSc
from dolfinx import fem

from projection import project
from solver_utils import sol_utils
import preconditioner

### useful infos for PETSc mats, vecs, solvers...
# https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.Mat-class.html
# https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.Vec-class.html
# https://www.mcs.anl.gov/petsc/documentation/faq.html
# https://www.mcs.anl.gov/petsc/documentation/linearsolvertable.html
# https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.KSP-class.html
# https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.PC-class.html

# standard nonlinear solver for FEM problems
class solver_nonlinear:

    def __init__(self, pb, solver_params={}):

        self.pb = pb[0] # currently only one monolithic problem considered

        # problem variables list
        self.x, self.is_ghosted = self.pb.get_problem_var_list()

        self.nfields = len(self.x)

        self.ptype = self.pb.problem_physics

        self.set_solver_params(solver_params)

        self.tolerances = {}
        for n in range(self.nfields):
            if len(self.tolres)>1: # if we have a list here, we need one tolerance per variable!
                self.tolerances['res'+str(n+1)] = self.tolres[n]
                self.tolerances['inc'+str(n+1)] = self.tolinc[n]
            else:
                self.tolerances['res'+str(n+1)] = self.tolres[0]
                self.tolerances['inc'+str(n+1)] = self.tolinc[0]

        self.initialize_petsc_solver()

        # sub-solver (for Lagrange-type constraints governed by a nonlinear system, e.g. 3D-0D coupling)
        if self.pb.sub_solve:
            self.subsol = solver_nonlinear_ode([self.pb.pb0], solver_params['subsolver_params'])
        else:
            self.subsol = None

        self.solutils = sol_utils(self)
        self.sepstring = self.solutils.timestep_separator()

        self.li_s = [] # linear iterations over all solves


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

        try: self.maxresval = solver_params['catch_max_res_value']
        except: self.maxresval = 1e16

        try: self.direct_solver = solver_params['direct_solver']
        except: self.direct_solver = 'mumps'

        try: self.iterative_solver = solver_params['iterative_solver']
        except: self.iterative_solver = 'gmres'

        try: self.precond_fields = solver_params['precond_fields']
        except: self.precond_fields = []

        try: self.fieldsplit_type = solver_params['fieldsplit_type']
        except: self.fieldsplit_type = 'jacobi'

        try: self.block_precond = solver_params['block_precond']
        except: self.block_precond = 'fieldsplit'

        try: self.tol_lin_rel = solver_params['tol_lin_rel']
        except: self.tol_lin_rel = 1.0e-5

        try: self.tol_lin_abs = solver_params['tol_lin_abs']
        except: self.tol_lin_abs = 1.0e-50

        try: self.res_lin_monitor = solver_params['res_lin_monitor']
        except: self.res_lin_monitor = 'rel'

        try: self.maxliniter = solver_params['max_liniter']
        except: self.maxliniter = 1200

        try: self.lin_norm_type = solver_params['lin_norm_type']
        except: self.lin_norm_type = 'unpreconditioned'

        if self.lin_norm_type=='preconditioned':
            self.linnormtype = 1
        elif self.lin_norm_type=='unpreconditioned':
            self.linnormtype = 2
        else:
            raise ValueError("Unknown lin_norm_type option!")

        try: self.print_liniter_every = solver_params['print_liniter_every']
        except: self.print_liniter_every = 1

        try: self.iset_options = solver_params['indexset_options']
        except: self.iset_options = {}
        is_option_keys = ['lms_to_p','lms_to_v','rom_to_new']
        # revert to defaults if not set by the user
        for k in is_option_keys:
            if k not in self.iset_options.keys(): self.iset_options[k] = False

        try: self.print_local_iter = solver_params['print_local_iter']
        except: self.print_local_iter = False

        try: self.tol_res_local = solver_params['tol_res_local']
        except: self.tol_res_local = 1.0e-10

        try: self.tol_inc_local = solver_params['tol_inc_local']
        except: self.tol_inc_local = 1.0e-10

        self.solvetype = solver_params['solve_type']

        # check if we have a list of tolerances (for coupled problems) or just one value
        self.tolres, self.tolinc = [], []
        if isinstance(solver_params['tol_res'], list):
            for n in range(len(solver_params['tol_res'])):
                self.tolres.append(solver_params['tol_res'][n])
                self.tolinc.append(solver_params['tol_inc'][n])
        else:
            self.tolres.append(solver_params['tol_res'])
            self.tolinc.append(solver_params['tol_inc'])


    def initialize_petsc_solver(self):

        # create solver
        self.ksp = PETSc.KSP().create(self.pb.comm)

        if self.solvetype=='direct':

            self.ksp.setType("preonly")
            self.ksp.getPC().setType("lu")
            self.ksp.getPC().setFactorSolverType(self.direct_solver)

        elif self.solvetype=='iterative':

            self.ksp.setInitialGuessNonzero(False)
            self.ksp.setNormType(self.linnormtype) # cf. https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.KSP.NormType-class.html

            # block iterative method
            if self.nfields > 1:

                self.ksp.setType(self.iterative_solver) # cf. https://petsc.org/release/petsc4py/petsc4py.PETSc.KSP.Type-class.html

                # TODO: how to use this adaptively...
                #self.ksp.getPC().setReusePreconditioner(True)

                if self.block_precond == 'fieldsplit':

                    # see e.g. https://petsc.org/main/manual/ksp/#sec-block-matrices
                    self.ksp.getPC().setType("fieldsplit")
                    # cf. https://petsc.org/main/manualpages/PC/PCCompositeType

                    if self.fieldsplit_type=='jacobi':
                        splittype = PETSc.PC.CompositeType.ADDITIVE # block Jacobi
                    elif self.fieldsplit_type=='gauss_seidel':
                        splittype = PETSc.PC.CompositeType.MULTIPLICATIVE # block Gauss-Seidel
                    elif self.fieldsplit_type=='gauss_seidel_sym':
                        splittype = PETSc.PC.CompositeType.SYMMETRIC_MULTIPLICATIVE # symmetric block Gauss-Seidel
                    elif self.fieldsplit_type=='schur':
                        assert(self.nfields==2)
                        splittype = PETSc.PC.CompositeType.SCHUR # block Schur - for 2x2 block systems only
                    else:
                        raise ValueError("Unknown fieldsplit_type option.")

                    self.ksp.getPC().setFieldSplitType(splittype)

                    iset = self.pb.get_index_sets(isoptions=self.iset_options)
                    nsets = len(iset)

                    # normally, nsets = self.nfields, but for a surface-projected ROM (FrSI) problem, we have one more index set than fields
                    if nsets==2:   self.ksp.getPC().setFieldSplitIS(("f1", iset[0]),("f2", iset[1]))
                    elif nsets==3: self.ksp.getPC().setFieldSplitIS(("f1", iset[0]),("f2", iset[1]),("f3", iset[2]))
                    elif nsets==4: self.ksp.getPC().setFieldSplitIS(("f1", iset[0]),("f2", iset[1]),("f3", iset[2]),("f4", iset[3]))
                    elif nsets==5: self.ksp.getPC().setFieldSplitIS(("f1", iset[0]),("f2", iset[1]),("f3", iset[2]),("f4", iset[3]),("f5", iset[4]))
                    else: raise RuntimeError("Currently, no more than 5 fields/index sets are supported.")

                    # get the preconditioners for each block
                    ksp_fields = self.ksp.getPC().getFieldSplitSubKSP()

                    assert(nsets==len(self.precond_fields)) # sanity check

                    # set field-specific preconditioners
                    for n in range(nsets):

                        if self.precond_fields[n]['prec'] == 'amg':
                            try: solvetype = self.precond_fields[n]['solve']
                            except: solvetype = "preonly"
                            ksp_fields[n].setType(solvetype)
                            try: amgtype = self.precond_fields[n]['amgtype']
                            except: amgtype = "hypre"
                            ksp_fields[n].getPC().setType(amgtype)
                            if amgtype=="hypre":
                                ksp_fields[n].getPC().setHYPREType("boomeramg")
                        elif self.precond_fields[n]['prec'] == 'direct':
                            ksp_fields[n].setType("preonly")
                            ksp_fields[n].getPC().setType("lu")
                            ksp_fields[n].getPC().setFactorSolverType("mumps")
                        else:
                            raise ValueError("Currently, only either 'amg' or 'direct' are supported as field-specific preconditioner.")

                elif self.block_precond == 'schur2x2':

                    self.ksp.getPC().setType(PETSc.PC.Type.PYTHON)
                    bj = preconditioner.schur_2x2(self.pb.get_index_sets(isoptions=self.iset_options),self.precond_fields,self.pb.comm)
                    self.ksp.getPC().setPythonContext(bj)

                elif self.block_precond == 'simple2x2':

                    self.ksp.getPC().setType(PETSc.PC.Type.PYTHON)
                    bj = preconditioner.simple_2x2(self.pb.get_index_sets(isoptions=self.iset_options),self.precond_fields,self.pb.comm)
                    self.ksp.getPC().setPythonContext(bj)

                elif self.block_precond == 'schur3x3':

                    self.ksp.getPC().setType(PETSc.PC.Type.PYTHON)
                    bj = preconditioner.schur_3x3(self.pb.get_index_sets(isoptions=self.iset_options),self.precond_fields,self.pb.comm)
                    self.ksp.getPC().setPythonContext(bj)

                elif self.block_precond == 'schur4x4':

                    self.ksp.getPC().setType(PETSc.PC.Type.PYTHON)
                    bj = preconditioner.schur_4x4(self.pb.get_index_sets(isoptions=self.iset_options),self.precond_fields,self.pb.comm)
                    self.ksp.getPC().setPythonContext(bj)

                elif self.block_precond == 'bgs2x2': # can also be called via PETSc's fieldsplit

                    self.ksp.getPC().setType(PETSc.PC.Type.PYTHON)
                    bj = preconditioner.bgs_2x2(self.pb.get_index_sets(isoptions=self.iset_options),self.precond_fields,self.pb.comm)
                    self.ksp.getPC().setPythonContext(bj)

                elif self.block_precond == 'jacobi2x2': # can also be called via PETSc's fieldsplit

                    self.ksp.getPC().setType(PETSc.PC.Type.PYTHON)
                    bj = preconditioner.jacobi_2x2(self.pb.get_index_sets(isoptions=self.iset_options),self.precond_fields,self.pb.comm)
                    self.ksp.getPC().setPythonContext(bj)

                else:
                    raise ValueError("Unknown block_precond option!")

            else:

                if self.precond_fields[0] == 'amg':
                    self.ksp.getPC().setType("hypre")
                    self.ksp.getPC().setMGLevels(3)
                    self.ksp.getPC().setHYPREType("boomeramg")
                else:
                    raise ValueError("Currently, only 'amg' is supported as single-field preconditioner.")

            # set tolerances and print routine
            self.ksp.setTolerances(rtol=self.tol_lin_rel, atol=self.tol_lin_abs, divtol=None, max_it=self.maxliniter)
            self.ksp.setMonitor(lambda ksp, its, rnorm: self.solutils.print_linear_iter(its,rnorm))

            # set some additional PETSc options
            petsc_options = PETSc.Options()
            petsc_options.setValue('-ksp_gmres_modifiedgramschmidt', True)
            self.ksp.setFromOptions()

        else:

            raise NameError("Unknown solvetype!")


    # solve for consistent initial acceleration a_old
    def solve_consistent_ini_acc(self, res_a, jac_aa, a_old):

        # create solver
        ksp = PETSc.KSP().create(self.pb.comm)

        if self.solvetype=='direct':
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType(self.direct_solver)
        elif self.solvetype=='iterative':
            ksp.setType(self.iterative_solver)
            ksp.getPC().setType("hypre")
            ksp.getPC().setMGLevels(3)
            ksp.getPC().setHYPREType("boomeramg")
        else:
            raise NameError("Unknown solvetype!")

        # solve for consistent initial acceleration a_old
        M_a = fem.petsc.assemble_matrix(jac_aa, [])
        M_a.assemble()

        r_a = fem.petsc.assemble_vector(res_a)
        r_a.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        ksp.setOperators(M_a)
        ksp.solve(-r_a, a_old.vector)

        a_old.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        r_a.destroy(), M_a.destroy()
        ksp.destroy()


    def solve_local(self, localdata):

        for l in range(len(localdata['var'])):
            self.newton_local(localdata['var'][l],localdata['res'][l],localdata['inc'][l],localdata['fnc'][l])


    def newton(self, t, localdata={}):

        # offset array for multi-field systems
        self.offsetarr = [0]
        off=0
        for n in range(self.nfields):
            if n==0:
                if self.pb.have_rom: # currently, ROM is only implemented for the first variable in the system!
                    off += self.pb.rom.V.getLocalSize()[1]
                else:
                    off += self.x[0].getLocalSize()
            else:
                off += self.x[n].getLocalSize()

            self.offsetarr.append(off)

        del_x, x_start = [[]]*self.nfields, [[]]*self.nfields

        for n in range(self.nfields):
            # solution increments for Newton
            del_x[n] = self.x[n].duplicate()
            del_x[n].set(0.0)
            # start vector (needed for reset of Newton in case of divergence)
            x_start[n] = self.x[n].duplicate()
            self.x[n].assemble()
            x_start[n].axpby(1.0, 0.0, self.x[n])
            if self.pb.sub_solve: # can only be a 0D model so far...
                s_start = self.pb.pb0.s.duplicate()
                self.pb.pb0.s.assemble()
                s_start.axpby(1.0, 0.0, self.pb.pb0.s)

        # Newton iteration index
        it = 0
        # for PTC
        k_PTC = self.k_PTC_initial
        counter_adapt, max_adapt = 0, 10
        self.ni, self.li = 0, 0 # nonlinear and linear iteration counters

        self.solutils.print_nonlinear_iter(header=True)

        while it < self.maxiter and counter_adapt < max_adapt:

            tes = time.time()

            if self.pb.localsolve:
                self.solve_local(localdata)

            r_list = self.pb.assemble_residual(t, subsolver=self.subsol)
            K_list = self.pb.assemble_stiffness(t, subsolver=self.subsol)

            if self.PTC:
                # computes K_00 + k_PTC * I
                K_list[0][0].shift(k_PTC)

            # model order reduction stuff - currently only on first mat in system...
            if self.pb.have_rom:
                # projection of main block: system matrix, residual, and increment
                tmp = K_list[0][0].matMult(self.pb.rom.V) # K_00 * V
                K_list[0][0] = self.pb.rom.V.transposeMatMult(tmp) # V^T * K_00 * V
                r_u_, del_u_ = self.pb.rom.V.createVecRight(), self.pb.rom.V.createVecRight()
                self.pb.rom.V.multTranspose(r_list[0], r_u_) # V^T * r_u
                # deal with penalties that may be added to reduced residual to penalize certain modes
                if bool(self.pb.rom.redbasisvec_penalties):
                    u_ = K_list[0][0].createVecRight()
                    self.pb.rom.V.multTranspose(self.x[0], u_) # V^T * u
                    penterm_ = self.pb.rom.V.createVecRight()
                    self.pb.rom.Cpen.mult(u_, penterm_) # Cpen * V^T * u
                    r_u_.axpy(1.0, penterm_) # add penalty term to reduced residual
                    K_list[0][0].aypx(1.0, self.pb.rom.CpenVTV) # K_00 + Cpen * V^T * V
                r_list[0].destroy(), del_x[0].destroy() # destroy, since we re-define the references!
                r_list[0], del_x[0] = r_u_, del_u_
                # now the offdiagonal blocks
                if self.nfields > 1:
                    for n in range(self.nfields-1):
                        if K_list[0][n+1] is not None:
                            K_list[0][n+1] = self.pb.rom.V.transposeMatMult(K_list[0][n+1]) # V^T * K_{0,n+1}
                        if K_list[n+1][0] is not None:
                            K_list[n+1][0] = K_list[n+1][0].matMult(self.pb.rom.V) # K_{n+1,0} * V

            te = time.time() - tes

            # we use a block matrix (either with merge-into-one or for a nested iterative solver) if we have more than one field
            if self.nfields > 1:

                tes = time.time()

                # nested variable vector
                r_full_nest = PETSc.Vec().createNest(r_list)

                # nested matrix
                K_full_nest = PETSc.Mat().createNest(K_list, isrows=None, iscols=None, comm=self.pb.comm)
                K_full_nest.assemble()

                te += time.time() - tes

                # for monolithic direct solver
                if self.solvetype=='direct':

                    tes = time.time()

                    K_full = PETSc.Mat()
                    K_full_nest.convert("aij", out=K_full)
                    K_full.assemble()

                    r_full = PETSc.Vec().createWithArray(r_full_nest.getArray())
                    r_full.assemble()

                    del_full = K_full.createVecLeft()
                    self.ksp.setOperators(K_full)
                    te += time.time() - tes

                    tss = time.time()
                    self.ksp.solve(-r_full, del_full)
                    ts = time.time() - tss

                # for nested iterative solver
                elif self.solvetype=='iterative':

                    tes = time.time()

                    # use same matrix as preconditioner
                    P_nest = K_full_nest

                    del_full = PETSc.Vec().createNest(del_x)

                    # if index sets do not align with the nested matrix structure
                    # anymore, we need a merged matrix to extract the submats
                    if self.iset_options['rom_to_new'] or self.iset_options['lms_to_p'] or self.iset_options['lms_to_v']:
                        P = PETSc.Mat()
                        P_nest.convert("aij", out=P)
                        P.assemble()
                        P_nest = P

                    self.ksp.setOperators(K_full_nest, P_nest)

                    r_full_nest.assemble()

                    # need to merge for non-fieldsplit-type preconditioners
                    if not self.block_precond == 'fieldsplit':
                        r_full = PETSc.Vec().createWithArray(r_full_nest.getArray())
                        r_full.assemble()
                        del_full = PETSc.Vec().createWithArray(del_full.getArray())
                        r_full_nest = r_full

                    te += time.time() - tes

                    tss = time.time()
                    self.ksp.solve(-r_full_nest, del_full)
                    ts = time.time() - tss

                    self.solutils.print_linear_iter_last(self.ksp.getIterationNumber(),self.ksp.getResidualNorm())

                else:

                    raise NameError("Unknown solvetype!")

                for n in range(self.nfields):
                    del_x[n].array[:] = del_full.array_r[self.offsetarr[n]:self.offsetarr[n+1]]

            else:

                # solve linear system
                self.ksp.setOperators(K_list[0][0])

                tss = time.time()
                self.ksp.solve(-r_list[0], del_x[0])
                ts = time.time() - tss

                if self.solvetype=='iterative':

                    self.solutils.print_linear_iter_last(self.ksp.getIterationNumber(),self.ksp.getResidualNorm())

            # get increment norm
            incnorms = {}
            for n in range(self.nfields):
                incnorms['inc'+str(n+1)] = del_x[n].norm()

            # reconstruct full-length increment vector - currently only for first var!
            if self.pb.have_rom:
                del_x[0] = self.pb.rom.V.createVecLeft()
                self.pb.rom.V.mult(del_u_, del_x[0]) # V * dx_red

            # update variables
            for n in range(self.nfields):
                self.x[n].axpy(1.0, del_x[n])
                if self.is_ghosted[n]==1:
                    self.x[n].ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                if self.is_ghosted[n]==2:
                    subvecs = self.x[n].getNestSubVecs()
                    for j in range(len(subvecs)): subvecs[j].ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            # get residual norm
            resnorms = {}
            for n in range(self.nfields):
                r_list[n].assemble()
                resnorms['res'+str(n+1)] = r_list[n].norm()

            self.solutils.print_nonlinear_iter(it,resnorms,incnorms,self.PTC,k_PTC,ts=ts,te=te)

            # destroy PETSc stuff...
            if self.nfields > 1:
                r_full_nest.destroy(), K_full_nest.destroy(), del_full.destroy()
                if self.solvetype=='direct': r_full.destroy(), K_full.destroy()
                if self.solvetype=='iterative': P_nest.destroy()
            if self.pb.have_rom:
                r_u_.destroy(), del_u_.destroy(), tmp.destroy()
            for n in range(self.nfields):
                r_list[n].destroy()
                for m in range(self.nfields):
                    if K_list[n][m] is not None: K_list[n][m].destroy()

            it += 1

            # for PTC - only applied to first main block so far...
            if self.PTC and it > 1 and res_norm_main_last > 0.: k_PTC *= resnorms['res1']/res_norm_main_last
            res_norm_main_last = resnorms['res1']

            # adaptive PTC (for 3D block K_00 only!)
            if self.divcont=='PTC':

                self.maxiter = 250
                err = self.solutils.catch_solver_errors(resnorms['res1'], incnorm=incnorms['inc1'], maxval=self.maxresval)

                if err:
                    self.PTC = True
                    # reset Newton step
                    it, k_PTC = 0, self.k_PTC_initial
                    if counter_adapt>0: k_PTC *= np.random.uniform(self.PTC_randadapt_range[0], self.PTC_randadapt_range[1])

                    if self.pb.comm.rank == 0:
                        print("PTC factor: %.4f" % (k_PTC))
                        sys.stdout.flush()

                    # reset solver
                    for n in range(self.nfields):
                        self.reset_step(self.x[n], x_start[n], self.is_ghosted[n])
                        if self.pb.sub_solve: # can only be a 0D model so far...
                            self.reset_step(self.pb.pb0.s, s_start, 0)

                    counter_adapt += 1

            # check if converged
            converged = self.solutils.check_converged(resnorms,incnorms,self.tolerances)
            if converged:
                # destroy PETSc vectors
                for n in range(self.nfields):
                    del_x[n].destroy(), x_start[n].destroy()
                if self.pb.sub_solve: s_start.destroy()
                # reset to normal Newton if PTC was used in a divcont action
                if self.divcont=='PTC':
                    self.PTC = False
                    counter_adapt = 0
                self.ni = it
                break

        else:

            raise RuntimeError("Newton did not converge after %i iterations!" % (it))


    def reset_step(self, vec, vec_start, ghosted):

        vec.axpby(1.0, 0.0, vec_start)

        if ghosted==1:
            vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        if ghosted==2:
            subvecs = vec.getNestSubVecs()
            for j in range(len(subvecs)): subvecs[j].ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


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



# solver for pure ODE (0D) problems (e.g. a system of first order ODEs integrated with One-Step-Theta method)
class solver_nonlinear_ode(solver_nonlinear):

    def __init__(self, pb, solver_params={}):

        self.pb = pb[0] # currently only one problem considered

        self.ptype = self.pb.problem_physics

        try: self.maxiter = solver_params['maxiter']
        except: self.maxiter = 25

        try: self.direct_solver = solver_params['direct_solver']
        except: self.direct_solver = 'mumps'

        self.tolres = solver_params['tol_res']
        self.tolinc = solver_params['tol_inc']

        self.tolerances = {'res1' : self.tolres, 'inc1' : self.tolinc}

        self.PTC = False # don't think we'll ever need PTC for the 0D ODE problem...
        self.solvetype = 'direct' # only a direct solver is available for ODE problems

        self.solutils = sol_utils(self)

        self.sepstring = self.solutils.timestep_separator()

        self.initialize_petsc_solver()


    def initialize_petsc_solver(self):

        # create solver
        self.ksp = PETSc.KSP().create(self.pb.comm)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")
        self.ksp.getPC().setFactorSolverType(self.direct_solver)


    def newton(self, t, print_iter=True, sub=False):

        # Newton iteration index
        it = 0

        if print_iter: self.solutils.print_nonlinear_iter(header=True,sub=sub)

        self.ni, self.li = 0, 0 # nonlinear and linear iteration counters (latter probably never relevant for ODE problems...)

        while it < self.maxiter:

            tes = time.time()

            self.pb.odemodel.evaluate(self.pb.s, t, self.pb.df, self.pb.f, self.pb.dK, self.pb.K, self.pb.c, self.pb.y, self.pb.aux)

            # ODE rhs vector and stiffness matrix
            r = self.pb.assemble_residual(t)
            K = self.pb.assemble_stiffness(t)

            ds = K.createVecLeft()

            # solve linear system
            self.ksp.setOperators(K)

            te = time.time() - tes

            tss = time.time()
            self.ksp.solve(-r, ds)
            ts = time.time() - tss

            # update solution
            self.pb.s.axpy(1.0, ds)

            # get norms
            inc_norm = ds.norm()
            #r = self.pb.assemble_residual(t)
            res_norm = r.norm()

            if print_iter: self.solutils.print_nonlinear_iter(it,{'res1' : res_norm},{'inc1' : inc_norm},ts=ts,te=te,sub=sub)

            # destroy PETSc stuff...
            ds.destroy(), r.destroy(), K.destroy()

            it += 1

            # check if converged
            converged = self.solutils.check_converged({'res1' : res_norm},{'inc1' : inc_norm},self.tolerances,ptype='flow0d')
            if converged:
                if print_iter and sub:
                    if self.pb.comm.rank == 0:
                        print('       ****************************************************\n')
                        sys.stdout.flush()
                self.ni = it
                break

        else:

            raise RuntimeError("Newton for ODE system did not converge after %i iterations!" % (it))
