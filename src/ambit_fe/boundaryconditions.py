#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import numpy as np
from dolfinx import fem, mesh
from petsc4py import PETSc
import ufl

from . import expression

"""
Boundary condition classes for all problems
"""

class boundary_cond():

    def __init__(self, io, fem_params=None, vf=None, ti=None, ki=None, ff=None, dim=None, V_field=None, Vdisc_scalar=None):

        self.io = io
        self.vf = vf
        self.ti = ti
        self.ki = ki
        self.ff = ff

        if dim is None:
            self.dim = self.io.mesh.topology.dim
        else:
            self.dim = dim

        if fem_params is not None:
            self.quad_degree = fem_params['quad_degree']

        # continuous function space of primary field (e.g. displacement or velocity)
        self.V_field = V_field
        # discontinuous scalar function space
        self.Vdisc_scalar = Vdisc_scalar

        self.dbcs = []

        self.have_dirichlet_file = False


    # set Dirichlet BCs (should probably be overloaded for problems that do not have vector variables...)
    def dirichlet_bcs(self, bcdict):

        for d in bcdict:

            codim = d.get('codimension', self.dim-1)

            if codim==self.dim-1: mdata = self.io.mt_b1
            if codim==self.dim-2: mdata = self.io.mt_b2
            if codim==self.dim-3: mdata = self.io.mt_b3

            func = fem.Function(self.V_field)

            if 'curve' in d.keys():
                assert('val' not in d.keys() and 'expression' not in d.keys() and 'file' not in d.keys())
                load = expression.template_vector(dim=self.dim)
                if d['dir'] == 'all': curve_x, curve_y, curve_z = d['curve'][0], d['curve'][1], d['curve'][2]
                else:                 curve_x, curve_y, curve_z = d['curve'], d['curve'], d['curve']
                load.val_x, load.val_y, load.val_z = self.ti.timecurves(curve_x)(self.ti.t_init), self.ti.timecurves(curve_y)(self.ti.t_init), self.ti.timecurves(curve_z)(self.ti.t_init)
                func.interpolate(load.evaluate)
                func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                self.ti.funcs_to_update_vec.append({func : [self.ti.timecurves(curve_x), self.ti.timecurves(curve_y), self.ti.timecurves(curve_z)]})
                self.ti.funcs_to_update_vec_old.append({None : -1}) # DBCs don't need an old state
            elif 'val' in d.keys():
                assert('curve' not in d.keys() and 'expression' not in d.keys() and 'file' not in d.keys())
                func.x.petsc_vec.set(d['val'])
            elif 'expression' in d.keys():
                assert('curve' not in d.keys() and 'val' not in d.keys() and 'file' not in d.keys())
                expr = d['expression']()
                expr.t = self.ti.t_init
                func.interpolate(expr.evaluate)
                func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                self.ti.funcsexpr_to_update_vec[func] = expr
                self.ti.funcsexpr_to_update_vec_old[func] = None # DBCs don't need an old state
            elif 'file' in d.keys():
                assert('curve' not in d.keys() and 'val' not in d.keys() and 'expression' not in d.keys())
                fle = d['file'] # a single file
                ftype = d.get('ftype', 'id_val')
                # to ramp the file by a time curve
                ramp_curve = d.get('ramp_curve', None)
                if ramp_curve is not None:
                    func_ramp, func_file = fem.Function(self.V_field), fem.Function(self.V_field)
                    # first read file into function
                    self.io.readfunction(func_file, fle, filetype=ftype)
                    # now store ramp curve into function
                    load_ = expression.template_vector(dim=self.dim)
                    load_.val_x, load_.val_y, load_.val_z = self.ti.timecurves(d['ramp_curve'])(self.ti.t_init), self.ti.timecurves(d['ramp_curve'])(self.ti.t_init), self.ti.timecurves(d['ramp_curve'])(self.ti.t_init)
                    func_ramp.interpolate(load_.evaluate)
                    func_ramp.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    self.ti.funcs_to_update_vec.append({func_ramp : [self.ti.timecurves(d['ramp_curve']), self.ti.timecurves(d['ramp_curve']), self.ti.timecurves(d['ramp_curve'])],
                                                        'funcs_mult' : [func_file, func]})
                    self.ti.funcs_to_update_vec_old.append({None : -1}) # DBCs don't need an old state
                    # now multiply
                    func.x.petsc_vec.pointwiseMult(func_ramp.x.petsc_vec, func_file.x.petsc_vec)
                    func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                else:
                    # read file into function
                    self.io.readfunction(func, fle, filetype=ftype)
            else:
                raise RuntimeError("Need to have 'curve', 'val', 'expression', or 'file' specified!")

            if d['dir'] == 'all':
                self.dbcs.append( fem.dirichletbc(func, fem.locate_dofs_topological(self.V_field, codim, mdata.indices[np.isin(mdata.values, d['id'])])) )

            elif d['dir'] == 'x':
                dofs_x = fem.locate_dofs_topological(self.V_field.sub(0), codim, mdata.indices[np.isin(mdata.values, d['id'])])
                self.dbcs.append( fem.dirichletbc(func.sub(0), dofs_x) )

            elif d['dir'] == 'y':
                dofs_y = fem.locate_dofs_topological(self.V_field.sub(1), codim, mdata.indices[np.isin(mdata.values, d['id'])])
                self.dbcs.append( fem.dirichletbc(func.sub(1), dofs_y) )

            elif d['dir'] == 'z':
                dofs_z = fem.locate_dofs_topological(self.V_field.sub(2), codim, mdata.indices[np.isin(mdata.values, d['id'])])
                self.dbcs.append( fem.dirichletbc(func.sub(2), dofs_z) )

            elif d['dir'] == '2dimX':
                dofs_x = fem.locate_dofs_topological(self.V_field.sub(0), codim, mesh.locate_entities_boundary(self.io.mesh, codim, self.twodimX))
                self.dbcs.append( fem.dirichletbc(func.sub(0), dofs_x) )

            elif d['dir'] == '2dimY':
                dofs_y = fem.locate_dofs_topological(self.V_field.sub(1), codim, mesh.locate_entities_boundary(self.io.mesh, codim, self.twodimY))
                self.dbcs.append( fem.dirichletbc(func.sub(1), dofs_y) )

            elif d['dir'] == '2dimZ':
                dofs_z = fem.locate_dofs_topological(self.V_field.sub(2), codim, mesh.locate_entities_boundary(self.io.mesh, codim, self.twodimZ))
                self.dbcs.append( fem.dirichletbc(func.sub(2), dofs_z) )

            else:
                raise NameError("Unknown dir option for Dirichlet BC!")


    def dirichlet_vol(self, bcdict):

        for d in bcdict:

            func = fem.Function(self.V_field)

            if 'curve' in d.keys():
                assert('val' not in d.keys() and 'file' not in d.keys())
                load = expression.template_vector(dim=self.dim)
                if d['dir'] == 'all': curve_x, curve_y, curve_z = d['curve'][0], d['curve'][1], d['curve'][2]
                else:                 curve_x, curve_y, curve_z = d['curve'], d['curve'], d['curve']
                load.val_x, load.val_y, load.val_z = self.ti.timecurves(curve_x)(self.ti.t_init), self.ti.timecurves(curve_y)(self.ti.t_init), self.ti.timecurves(curve_z)(self.ti.t_init)
                func.interpolate(load.evaluate)
                func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                self.ti.funcs_to_update_vec.append({func : [self.ti.timecurves(curve_x), self.ti.timecurves(curve_y), self.ti.timecurves(curve_z)]})
                self.ti.funcs_to_update_vec_old.append({None : -1}) # DBCs don't need an old state
            elif 'val' in d.keys():
                assert('curve' not in d.keys() and 'file' not in d.keys())
                func.x.petsc_vec.set(d['val'])
            elif 'file' in d.keys(): # file series, where we'd have one file per time step
                assert('val' not in d.keys() and 'curve' not in d.keys())
                scale = d.get('scale', 1.)
                self.ti.funcs_data.append({func : d['file'], 'scale' : scale})
                self.have_dirichlet_file = True
            else:
                raise RuntimeError("Need to have 'curve', 'val', or 'file' specified!")

            self.dbcs.append( fem.dirichletbc(func, fem.locate_dofs_topological(self.V_field, self.dim, self.io.mt_d.indices[np.isin(self.io.mt_d.values, d['id'])])) )


    # function to mark x=0
    def twodimX(self, x):
        return np.isclose(x[0], 0.0)

    # function to mark y=0
    def twodimY(self, x):
        return np.isclose(x[1], 0.0)

    # function to mark z=0
    def twodimZ(self, x):
        return np.isclose(x[2], 0.0)


    # set Neumann BCs
    def neumann_bcs(self, bcdict, ds_, F=None, funcs_to_update=None, funcs_to_update_vec=None, funcsexpr_to_update=None, funcsexpr_to_update_vec=None):

        w = ufl.as_ufl(0)

        for n in bcdict:

            codim = n.get('codimension', self.dim-1)

            if codim==self.dim-1: dind=0
            elif codim==self.dim-2: dind=1
            else: raise ValueError("Wrong codimension of boundary.")

            if n['dir'] == 'xyz_ref': # reference xyz

                func = fem.Function(self.V_field)

                if 'curve' in n.keys():
                    assert('val' not in n.keys() and 'expression' not in n.keys())
                    load = expression.template_vector(dim=self.dim)
                    load.val_x, load.val_y, load.val_z = self.ti.timecurves(n['curve'][0])(self.ti.t_init), self.ti.timecurves(n['curve'][1])(self.ti.t_init), self.ti.timecurves(n['curve'][2])(self.ti.t_init)
                    func.interpolate(load.evaluate)
                    func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    funcs_to_update_vec.append({func : [self.ti.timecurves(n['curve'][0]), self.ti.timecurves(n['curve'][1]), self.ti.timecurves(n['curve'][2])]})
                elif 'val' in n.keys():
                    assert('curve' not in n.keys() and 'expression' not in n.keys())
                    func.x.petsc_vec.set(n['val']) # currently only one value for all directions - use constant load function otherwise!
                elif 'expression' in n.keys():
                    assert('curve' not in n.keys() and 'val' not in n.keys())
                    expr = n['expression']()
                    expr.t = self.ti.t_init
                    func.interpolate(expr.evaluate)
                    func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    funcsexpr_to_update_vec[func] = expr
                else:
                    raise RuntimeError("Need to have 'curve', 'val', or 'expression' specified!")

                for i in range(len(n['id'])):

                    w += self.vf.deltaW_ext_neumann_ref(func, ds_[dind](n['id'][i]))

            elif n['dir'] == 'normal_ref': # reference normal

                func = fem.Function(self.Vdisc_scalar)

                if 'curve' in n.keys():
                    assert('val' not in n.keys() and 'expression' not in n.keys())
                    load = expression.template()
                    load.val = self.ti.timecurves(n['curve'])(self.ti.t_init)
                    func.interpolate(load.evaluate)
                    func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    funcs_to_update.append({func : self.ti.timecurves(n['curve'])})
                elif 'val' in n.keys():
                    assert('curve' not in n.keys() and 'expression' not in n.keys())
                    func.x.petsc_vec.set(n['val'])
                elif 'expression' in n.keys():
                    assert('curve' not in n.keys() and 'val' not in n.keys())
                    expr = n['expression']()
                    expr.t = self.ti.t_init
                    func.interpolate(expr.evaluate)
                    func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    funcsexpr_to_update[func] = expr
                else:
                    raise RuntimeError("Need to have 'curve', 'val', or 'expression' specified!")

                for i in range(len(n['id'])):

                    w += self.vf.deltaW_ext_neumann_normal_ref(func, ds_[dind](n['id'][i]))

            elif n['dir'] == 'xyz_cur': # current xyz

                func = fem.Function(self.V_field)

                if 'curve' in n.keys():
                    assert('val' not in n.keys() and 'expression' not in n.keys())
                    load = expression.template_vector(dim=self.dim)
                    load.val_x, load.val_y, load.val_z = self.ti.timecurves(n['curve'][0])(self.ti.t_init), self.ti.timecurves(n['curve'][1])(self.ti.t_init), self.ti.timecurves(n['curve'][2])(self.ti.t_init)
                    func.interpolate(load.evaluate)
                    func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    funcs_to_update_vec.append({func : [self.ti.timecurves(n['curve'][0]), self.ti.timecurves(n['curve'][1]), self.ti.timecurves(n['curve'][2])]})
                elif 'val' in n.keys():
                    assert('curve' not in n.keys() and 'expression' not in n.keys())
                    func.x.petsc_vec.set(n['val']) # currently only one value for all directions - use constant load function otherwise!
                elif 'expression' in n.keys():
                    assert('curve' not in n.keys() and 'val' not in n.keys())
                    expr = n['expression']()
                    expr.t = self.ti.t_init
                    func.interpolate(expr.evaluate)
                    func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    funcsexpr_to_update_vec[func] = expr
                else:
                    raise RuntimeError("Need to have 'curve', 'val', or 'expression' specified!")

                for i in range(len(n['id'])):

                    w += self.vf.deltaW_ext_neumann_cur(func, ds_[dind](n['id'][i]), F=F)

            elif n['dir'] == 'normal_cur': # current normal

                func = fem.Function(self.Vdisc_scalar)

                if 'curve' in n.keys():
                    assert('val' not in n.keys() and 'expression' not in n.keys())
                    load = expression.template()
                    load.val = self.ti.timecurves(n['curve'])(self.ti.t_init)
                    func.interpolate(load.evaluate)
                    func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    funcs_to_update.append({func : self.ti.timecurves(n['curve'])})
                elif 'val' in n.keys():
                    assert('curve' not in n.keys() and 'expression' not in n.keys())
                    func.x.petsc_vec.set(n['val'])
                elif 'expression' in n.keys():
                    assert('curve' not in n.keys() and 'val' not in n.keys())
                    expr = n['expression']()
                    expr.t = self.ti.t_init
                    func.interpolate(expr.evaluate)
                    func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    funcsexpr_to_update[func] = expr
                else:
                    raise RuntimeError("Need to have 'curve', 'val', or 'expression' specified!")

                for i in range(len(n['id'])):

                    w += self.vf.deltaW_ext_neumann_normal_cur(func, ds_[dind](n['id'][i]), F=F)

            else:
                raise NameError("Unknown dir option for Neumann BC!")

        return w


    # set Neumann BCs for prestress
    def neumann_prestress_bcs(self, bcdict, ds_, funcs_to_update=None, funcs_to_update_vec=None, funcsexpr_to_update=None, funcsexpr_to_update_vec=None):

        w = ufl.as_ufl(0)

        for n in bcdict:

            codim = n.get('codimension', self.dim-1)

            if codim==self.dim-1: dind=0
            elif codim==self.dim-2: dind=1
            else: raise ValueError("Wrong codimension of boundary.")

            if n['dir'] == 'xyz_ref': # reference xyz

                func = fem.Function(self.V_field)

                if 'curve' in n.keys():
                    assert('val' not in n.keys() and 'expression' not in n.keys())
                    load = expression.template_vector(dim=self.dim)
                    load.val_x, load.val_y, load.val_z = self.ti.timecurves(n['curve'][0])(self.ti.t_init), self.ti.timecurves(n['curve'][1])(self.ti.t_init), self.ti.timecurves(n['curve'][2])(self.ti.t_init)
                    func.interpolate(load.evaluate)
                    func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    funcs_to_update_vec.append({func : [self.ti.timecurves(n['curve'][0]), self.ti.timecurves(n['curve'][1]), self.ti.timecurves(n['curve'][2])]})
                elif 'val' in n.keys():
                    assert('curve' not in n.keys() and 'expression' not in n.keys())
                    func.x.petsc_vec.set(n['val']) # currently only one value for all directions - use constant load function otherwise!
                elif 'expression' in n.keys():
                    assert('curve' not in n.keys() and 'val' not in n.keys())
                    expr = n['expression']()
                    expr.t = self.ti.t_init
                    func.interpolate(expr.evaluate)
                    func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    funcsexpr_to_update_vec[func] = expr
                else:
                    raise RuntimeError("Need to have 'curve', 'val', or 'expression' specified!")

                for i in range(len(n['id'])):

                    w += self.vf.deltaW_ext_neumann_ref(func, ds_[dind](n['id'][i]))

            elif n['dir'] == 'normal_ref': # reference normal

                func = fem.Function(self.Vdisc_scalar)

                if 'curve' in n.keys():
                    assert('val' not in n.keys() and 'expression' not in n.keys())
                    load = expression.template()
                    load.val = self.ti.timecurves(n['curve'])(self.ti.t_init)
                    func.interpolate(load.evaluate)
                    func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    funcs_to_update.append({func : self.ti.timecurves(n['curve'])})
                elif 'val' in n.keys():
                    assert('curve' not in n.keys() and 'expression' not in n.keys())
                    func.x.petsc_vec.set(n['val']) # currently only one value for all directions - use constant load function otherwise!
                elif 'expression' in n.keys():
                    assert('curve' not in n.keys() and 'val' not in n.keys())
                    expr = n['expression']()
                    expr.t = self.ti.t_init
                    func.interpolate(expr.evaluate)
                    func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                    funcsexpr_to_update[func] = expr
                else:
                    raise RuntimeError("Need to have 'curve', 'val', or 'expression' specified!")

                for i in range(len(n['id'])):

                    w += self.vf.deltaW_ext_neumann_normal_ref(func, ds_[dind](n['id'][i]))

            else:
                raise NameError("Unknown dir option for Neumann prestress BC!")

        return w


    # set Robin BCs
    def robin_bcs(self, bcdict, u, v, ds_, u_pre=None, wel=None, F=None):

        w = ufl.as_ufl(0)

        if wel is None:
            wel_ = ufl.constantvalue.zero(self.dim)
        else:
            wel_ = wel

        for r in bcdict:

            codim = r.get('codimension', self.dim-1)

            if codim==self.dim-1: dind=0
            elif codim==self.dim-2: dind=1
            else: raise ValueError("Wrong codimension of boundary.")

            if r['type'] == 'spring':

                # may be an expression
                stiff = r['stiff']
                if inspect.isclass(stiff):
                    stiff_expr = stiff()
                    stiff_ = fem.Function(self.Vdisc_scalar)
                    stiff_.interpolate(stiff_expr.evaluate)
                else:
                    stiff_ = stiff

                if r['dir'] == 'xyz_ref': # reference xyz

                    for i in range(len(r['id'])):

                        w += self.vf.deltaW_ext_robin_spring(u, stiff_, ds_[dind](r['id'][i]), u_pre)

                elif r['dir'] == 'normal_ref': # reference normal

                    for i in range(len(r['id'])):

                        w += self.vf.deltaW_ext_robin_spring_normal_ref(u, stiff_, ds_[dind](r['id'][i]), u_pre)

                elif r['dir'] == 'normal_cross': # cross normal

                    for i in range(len(r['id'])):

                        w += self.vf.deltaW_ext_robin_spring_normal_cross(u, stiff_, ds_[dind](r['id'][i]), u_pre)

                else:
                    raise NameError("Unknown dir option for Robin BC!")


            elif r['type'] == 'dashpot':

                # may be an expression
                visc = r['visc']
                if inspect.isclass(visc):
                    visc_expr = visc()
                    visc_ = fem.Function(self.Vdisc_scalar)
                    visc_.interpolate(visc_expr.evaluate)
                else:
                    visc_ = visc

                if r['dir'] == 'xyz_ref': # reference xyz

                    for i in range(len(r['id'])):

                        w += self.vf.deltaW_ext_robin_dashpot(v, visc_, ds_[dind](r['id'][i]))

                elif r['dir'] == 'normal_ref': # reference normal

                    for i in range(len(r['id'])):

                        w += self.vf.deltaW_ext_robin_dashpot_normal_ref(v, visc_, ds_[dind](r['id'][i]))

                elif r['dir'] == 'normal_cross': # cross normal

                    for i in range(len(r['id'])):

                        w += self.vf.deltaW_ext_robin_dashpot_normal_cross(v, visc_, ds_[dind](r['id'][i]))

                else:
                    raise NameError("Unknown dir option for Robin BC!")


            else:
                raise NameError("Unknown type option for Robin BC!")

        return w


    # set membrane surface BCs
    def membranesurf_bcs(self, bcdict, u, v, a, ds_, ivar=None, wallfields=[], actweights=[]):

        w, idmem, bstress, bstrainenergy, bintpower = ufl.as_ufl(0), [], [], [], []

        mi=0
        for m in bcdict:

            codim = m.get('codimension', self.dim-1)

            internal = m.get('internal', False)

            if codim==self.dim-1: dind=0
            elif codim==self.dim-2: dind=1
            else: raise ValueError("Wrong codimension of boundary.")

            if internal:
                dind=2
                fcts = m.get('facet_side', '+')
            else:
                fcts = None

            # field for variable wall thickness
            if bool(wallfields):
                wallfield = wallfields[mi]
            else:
                wallfield = None

            # field for active stress weighting
            if bool(actweights):
                actweight = actweights[mi]
            else:
                actweight = None

            for i in range(len(m['id'])):

                idmem.append(m['id'][i])

                w += self.vf.deltaW_ext_membrane(self.ki.F(u), self.ki.Fdot(v), a, m['params'], ds_[dind](m['id'][i]), ivar=ivar, fibfnc=self.ff, wallfield=wallfield, actweight=actweight, fcts=fcts)
                bstr, bse, bip = self.vf.deltaW_ext_membrane(self.ki.F(u), self.ki.Fdot(v), a, m['params'], ds_[dind](m['id'][i]), ivar=ivar, fibfnc=self.ff, wallfield=wallfield, actweight=actweight, fcts=fcts, returnquantity='stress_energy_power')
                bstress.append(bstr)
                bstrainenergy.append(bse)
                bintpower.append(bip)

            mi+=1

        return w, idmem, bstress, bstrainenergy, bintpower


    # set body forces (technically, no "boundary" conditions, since acting on a volume element... but implemented here for convenience)
    def bodyforce(self, bcdict, dx_, funcs_to_update=None, funcsexpr_to_update=None):

        w = ufl.as_ufl(0)

        for b in bcdict:

            func, func_dir = fem.Function(self.Vdisc_scalar), fem.Function(self.V_field)

            # direction needs to be set
            driection = expression.template_vector(dim=self.dim)
            dir_x, dir_y, dir_z = b['dir'][0], b['dir'][1], b['dir'][2]
            dir_norm = np.sqrt(dir_x**2. + dir_y**2. + dir_z**2.)
            driection.val_x, driection.val_y, driection.val_z = dir_x/dir_norm, dir_y/dir_norm, dir_z/dir_norm
            func_dir.interpolate(driection.evaluate)

            if 'curve' in b.keys():
                assert('val' not in b.keys() and 'expression' not in b.keys())
                load = expression.template()
                load.val = self.ti.timecurves(b['curve'])(self.ti.t_init)
                func.interpolate(load.evaluate)
                func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                funcs_to_update.append({func : self.ti.timecurves(b['curve'])})
            elif 'val' in b.keys():
                assert('curve' not in b.keys() and 'expression' not in b.keys())
                func.x.petsc_vec.set(b['val'])
            elif 'expression' in b.keys():
                assert('curve' not in b.keys() and 'val' not in b.keys())
                expr = b['expression']()
                expr.t = self.ti.t_init
                func.interpolate(expr.evaluate)
                func.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                funcsexpr_to_update[func] = expr
            else:
                raise RuntimeError("Need to have 'curve', 'val', or 'expression' specified!")

            for i in range(len(b['id'])):

                w += self.vf.deltaW_ext_bodyforce(func, func_dir, dx_(b['id'][i]))

        return w


class boundary_cond_fluid(boundary_cond):

    # set stabilized Neumann BCs
    def stabilized_neumann_bcs(self, bcdict, v, ds_, wel=None, F=None):

        w = ufl.as_ufl(0)

        for sn in bcdict:

            codim = sn.get('codimension', self.dim-1)

            if codim==self.dim-1: dind=0
            elif codim==self.dim-2: dind=1
            else: raise ValueError("Wrong codimension of boundary.")

            for i in range(len(sn['id'])):

                beta = sn['beta']

                w += self.vf.deltaW_ext_stabilized_neumann(v, beta, ds_[dind](sn['id'][i]), w=wel, F=F)

        return w


    # set mod. stabilized Neumann BCs
    def stabilized_neumann_mod_bcs(self, bcdict, v, ds_, wel=None, F=None):

        w = ufl.as_ufl(0)

        for sn in bcdict:

            codim = sn.get('codimension', self.dim-1)

            if codim==self.dim-1: dind=0
            elif codim==self.dim-2: dind=1
            else: raise ValueError("Wrong codimension of boundary.")

            for i in range(len(sn['id'])):

                beta = sn['beta']
                gamma = sn['gamma']

                w += self.vf.deltaW_ext_stabilized_neumann_mod(v, beta, gamma, ds_[dind](sn['id'][i]), w=wel, F=F)

        return w


    # set Robin valve BCs
    def robin_valve_bcs(self, bcdict, v, beta_, dS_, wel=None, F=None, dw=None):

        w = ufl.as_ufl(0)

        if wel is None:
            wel_ = ufl.constantvalue.zero(self.dim)
        else:
            wel_ = wel

        for r in bcdict:

            dwddp = ufl.as_ufl(0)

            codim = r.get('codimension', self.dim-1)

            direction = r.get('dir', 'xyz_ref')

            if codim==self.dim-1: dind=0
            elif codim==self.dim-2: dind=1
            else: raise ValueError("Wrong codimension of boundary.")

            beta_.append( fem.Function(self.Vdisc_scalar) )

            if direction == 'xyz_ref': # reference xyz

                for i in range(len(r['id'])):

                    if dw is None:
                        w += self.vf.deltaW_ext_robin_valve(v, beta_[-1], dS_[dind](r['id'][i]), fcts='+', w=wel_, F=F)
                    else:
                        # derivative (for implicit valve law)
                        dwddp += self.vf.deltaW_ext_robin_valve(v, ufl.as_ufl(1.0), dS_[dind](r['id'][i]), fcts='+', w=wel_, F=F)

            elif direction == 'normal_ref': # reference normal

                for i in range(len(r['id'])):

                    if dw is None:
                        w += self.vf.deltaW_ext_robin_valve_normal_ref(v, beta_[-1], dS_[dind](r['id'][i]), fcts='+', w=wel_, F=F)
                    else:
                        # derivative (for implicit valve law)
                        dwddp += self.vf.deltaW_ext_robin_valve_normal_ref(v, ufl.as_ufl(1.0), dS_[dind](r['id'][i]), fcts='+', w=wel_, F=F)

            else:
                raise NameError("Unknown dir option for Robin valve BC!")

            # one dwddp term per valve
            if dw is not None:
                dw.append( dwddp )

        return w


    # set flux monitor conditions
    def flux_monitor_bcs(self, bcdict, v, qdict_, wel=None, F=None):

        if wel is None:
            wel_ = ufl.constantvalue.zero(self.dim)
        else:
            wel_ = wel

        for r in bcdict:

            codim = r.get('codimension', self.dim-1)

            if codim==self.dim-1: dind=0
            elif codim==self.dim-2: dind=1
            else: raise ValueError("Wrong codimension of boundary.")

            q = ufl.as_ufl(0)

            internal = r.get('internal', False)
            on_subdomain = r.get('on_subdomain', False)

            if internal:
                assert(not on_subdomain)
                fcts = r.get('facet_side', '+')
            else:
                fcts = None

            if on_subdomain:
                dom_u = r['domain']

            for i in range(len(r['id'])):

                if not internal:
                    if not on_subdomain:
                        db_ = ufl.ds(domain=self.io.mesh_master, subdomain_data=self.io.mt_b1_master, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})
                    else:
                        db_ = ufl.ds(domain=self.io.submshes_emap[dom_u][0], subdomain_data=self.io.sub_mt_b1[dom_u], subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})
                else:
                    db_ = ufl.dS(domain=self.io.mesh_master, subdomain_data=self.io.mt_b1_master, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})

                q += self.vf.flux(v, db_, w=wel_, F=F, fcts=fcts)

            if on_subdomain:
                # entity map child to parent
                em_u = [self.io.submshes_emap[dom_u][1]]
                qdict_.append( fem.form(q, entity_maps=em_u) )
            else:
                qdict_.append( fem.form(q) )


    # set dp monitor conditions
    def dp_monitor_bcs(self, bcdict, a_u_, a_d_, pint_u_, pint_d_, pdict, F=None):

        for r in bcdict:

            codim = r.get('codimension', self.dim-1)

            spatial = r.get('spatial', False)

            if codim==self.dim-1: dind=0
            elif codim==self.dim-2: dind=1
            else: raise ValueError("Wrong codimension of boundary.")

            # area map for integration
            if spatial and F is not None:
                J = ufl.det(F)
                ja = J*ufl.sqrt(ufl.dot(self.vf.n0, (ufl.inv(F)*ufl.inv(F).T)*self.vf.n0))
            else:
                ja = 1.0

            dom_u, dom_d = r['upstream_domain'], r['downstream_domain']

            a_u, a_d, pint_u, pint_d = ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0), ufl.as_ufl(0)

            for i in range(len(r['id'])):

                db_u_ = ufl.ds(domain=self.io.submshes_emap[dom_u][0], subdomain_data=self.io.sub_mt_b1[dom_u], subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})
                db_d_ = ufl.ds(domain=self.io.submshes_emap[dom_d][0], subdomain_data=self.io.sub_mt_b1[dom_d], subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})

                # area forms
                a_u += ja*db_u_
                a_d += ja*db_d_

                # pressure forms
                pint_u += pdict[dom_u]*ja*db_u_
                pint_d += pdict[dom_d]*ja*db_d_

            # entity maps child to parent
            em_u = [self.io.submshes_emap[dom_u][1]]
            em_d = [self.io.submshes_emap[dom_d][1]]

            a_u_.append( fem.form(a_u, entity_maps=em_u) )
            a_d_.append( fem.form(a_d, entity_maps=em_d) )

            pint_u_.append( fem.form(pint_u, entity_maps=em_u) )
            pint_d_.append( fem.form(pint_d, entity_maps=em_d) )
