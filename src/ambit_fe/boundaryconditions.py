#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from dolfinx import fem, mesh
import ufl

from . import expression


class boundary_cond():

    def __init__(self, fem_params, io, vf, ti, ki=None, ff=None):

        self.io = io
        self.vf = vf
        self.ti = ti
        self.ki = ki
        self.ff = ff

        self.quad_degree = fem_params['quad_degree']

        self.dbcs = []

        self.have_dirichlet_file = False


    # set Dirichlet BCs (should probably be overloaded for problems that do not have vector variables...)
    def dirichlet_bcs(self, bcdict, V):

        for d in bcdict:

            try: bdim_r = d['bdim_reduction']
            except: bdim_r = 1

            if bdim_r==1: mdata = self.io.mt_b1
            if bdim_r==2: mdata = self.io.mt_b2
            if bdim_r==3: mdata = self.io.mt_b3

            func = fem.Function(V)

            if 'curve' in d.keys():
                assert('val' not in d.keys())
                load = expression.template_vector()
                if d['dir'] == 'all': curve_x, curve_y, curve_z = d['curve'][0], d['curve'][1], d['curve'][2]
                else:                 curve_x, curve_y, curve_z = d['curve'], d['curve'], d['curve']
                load.val_x, load.val_y, load.val_z = self.ti.timecurves(curve_x)(self.ti.t_init), self.ti.timecurves(curve_y)(self.ti.t_init), self.ti.timecurves(curve_z)(self.ti.t_init)
                func.interpolate(load.evaluate)
                self.ti.funcs_to_update_vec.append({func : [self.ti.timecurves(curve_x), self.ti.timecurves(curve_y), self.ti.timecurves(curve_z)]})
            elif 'val' in d.keys():
                assert('curve' not in d.keys())
                func.vector.set(d['val'])
            else:
                raise RuntimeError("Need to have 'curve' or 'val' specified!")

            if d['dir'] == 'all':
                for i in range(len(d['id'])):
                    self.dbcs.append( fem.dirichletbc(func, fem.locate_dofs_topological(V, self.io.mesh.topology.dim-bdim_r, mdata.indices[mdata.values == d['id'][i]])) )

            elif d['dir'] == 'x':
                for i in range(len(d['id'])):
                    dofs_x = fem.locate_dofs_topological(V.sub(0), self.io.mesh.topology.dim-bdim_r, mdata.indices[mdata.values == d['id'][i]])
                    self.dbcs.append( fem.dirichletbc(func.sub(0), dofs_x) )

            elif d['dir'] == 'y':
                for i in range(len(d['id'])):
                    dofs_y = fem.locate_dofs_topological(V.sub(1), self.io.mesh.topology.dim-bdim_r, mdata.indices[mdata.values == d['id'][i]])
                    self.dbcs.append( fem.dirichletbc(func.sub(1), dofs_y) )

            elif d['dir'] == 'z':
                for i in range(len(d['id'])):
                    dofs_z = fem.locate_dofs_topological(V.sub(2), self.io.mesh.topology.dim-bdim_r, mdata.indices[mdata.values == d['id'][i]])
                    self.dbcs.append( fem.dirichletbc(func.sub(2), dofs_z) )

            elif d['dir'] == '2dimX':
                dofs_x = fem.locate_dofs_topological(V.sub(0), self.io.mesh.topology.dim-bdim_r, mesh.locate_entities_boundary(self.io.mesh, self.io.mesh.topology.dim-bdim_r, self.twodimX))
                self.dbcs.append( fem.dirichletbc(func.sub(0), dofs_x) )

            elif d['dir'] == '2dimY':
                dofs_y = fem.locate_dofs_topological(V.sub(1), self.io.mesh.topology.dim-bdim_r, mesh.locate_entities_boundary(self.io.mesh, self.io.mesh.topology.dim-bdim_r, self.twodimY))
                self.dbcs.append( fem.dirichletbc(func.sub(1), dofs_y) )

            elif d['dir'] == '2dimZ':
                dofs_z = fem.locate_dofs_topological(V.sub(2), self.io.mesh.topology.dim-bdim_r, mesh.locate_entities_boundary(self.io.mesh, self.io.mesh.topology.dim-bdim_r, self.twodimZ))
                self.dbcs.append( fem.dirichletbc(func.sub(2), dofs_z) )

            else:
                raise NameError("Unknown dir option for Dirichlet BC!")


    def dirichlet_vol(self, bcdict, V):

        for d in bcdict:

            func = fem.Function(V)

            if 'curve' in d.keys():
                assert('val' not in d.keys())
                assert('file' not in d.keys())
                load = expression.template_vector()
                if d['dir'] == 'all': curve_x, curve_y, curve_z = d['curve'][0], d['curve'][1], d['curve'][2]
                else:                 curve_x, curve_y, curve_z = d['curve'], d['curve'], d['curve']
                load.val_x, load.val_y, load.val_z = self.ti.timecurves(curve_x)(self.ti.t_init), self.ti.timecurves(curve_y)(self.ti.t_init), self.ti.timecurves(curve_z)(self.ti.t_init)
                func.interpolate(load.evaluate)
                self.ti.funcs_to_update_vec.append({func : [self.ti.timecurves(curve_x), self.ti.timecurves(curve_y), self.ti.timecurves(curve_z)]})
            elif 'val' in d.keys():
                assert('curve' not in d.keys())
                assert('file' not in d.keys())
                func.vector.set(d['val'])
            elif 'file' in d.keys():
                assert('val' not in d.keys())
                assert('curve' not in d.keys())
                self.ti.funcs_data.append({func : d['file']})
                self.have_dirichlet_file = True
            else:
                raise RuntimeError("Need to have 'curve', 'val', or 'file' specified!")

            for i in range(len(d['id'])):
                self.dbcs.append( fem.dirichletbc(func, fem.locate_dofs_topological(V, self.io.mesh.topology.dim, self.io.mt_d.indices[self.io.mt_d.values == d['id'][i]])) )


    # function to mark x=0
    def twodimX(self, x):
        return np.isclose(x[0], 0.0)

    # function to mark y=0
    def twodimY(self, x):
        return np.isclose(x[1], 0.0)

    # function to mark z=0
    def twodimZ(self, x):
        return np.isclose(x[2], 0.0)


    # set Robin BCs
    def robin_bcs(self, bcdict, u, v, u_pre=None):

        w = ufl.as_ufl(0)

        for r in bcdict:

            try: bdim_r = r['bdim_reduction']
            except: bdim_r = 1

            if bdim_r==1: mdata = self.io.mt_b1_master
            if bdim_r==2: mdata = self.io.mt_b2_master
            if bdim_r==3: mdata = self.io.mt_b3_master

            if r['type'] == 'spring':

                if r['dir'] == 'xyz_ref': # reference xyz

                    for i in range(len(r['id'])):

                        db_ = ufl.ds(domain=self.io.mesh_master, subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})

                        w += self.vf.deltaW_ext_robin_spring(u, r['stiff'], db_, u_pre)

                elif r['dir'] == 'normal_ref': # reference normal

                    for i in range(len(r['id'])):

                        db_ = ufl.ds(domain=self.io.mesh_master, subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})

                        w += self.vf.deltaW_ext_robin_spring_normal_ref(u, r['stiff'], db_, u_pre)

                else:
                    raise NameError("Unknown dir option for Robin BC!")


            elif r['type'] == 'dashpot':

                if r['dir'] == 'xyz_ref':

                    for i in range(len(r['id'])):

                        db_ = ufl.ds(domain=self.io.mesh_master, subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})

                        w     += self.vf.deltaW_ext_robin_dashpot(v, r['visc'], db_)

                elif r['dir'] == 'normal_ref': # reference normal

                    for i in range(len(r['id'])):

                        db_ = ufl.ds(domain=self.io.mesh_master, subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})

                        w += self.vf.deltaW_ext_robin_dashpot_normal_ref(v, r['visc'], db_)

                else:
                    raise NameError("Unknown dir option for Robin BC!")


            else:
                raise NameError("Unknown type option for Robin BC!")

        return w


    # set membrane surface BCs
    def membranesurf_bcs(self, bcdict, u, v, a, varu, ivar=None, wallfields=[]):

        w, db_, bstress = ufl.as_ufl(0), [], []

        mi=0
        for m in bcdict:

            try: bdim_r = m['bdim_reduction']
            except: bdim_r = 1

            if bdim_r==1: mdata = self.io.mt_b1_master
            if bdim_r==2: mdata = self.io.mt_b2_master
            if bdim_r==3: mdata = self.io.mt_b3_master

            # field for variable wall thickness
            if bool(wallfields):
                wallfield = wallfields[mi]
            else:
                wallfield = None

            for i in range(len(m['id'])):

                db_.append(ufl.ds(domain=self.io.mesh_master, subdomain_data=mdata, subdomain_id=m['id'][i], metadata={'quadrature_degree': self.quad_degree}))

                w += self.vf.deltaW_ext_membrane(self.ki.F(u), self.ki.Fdot(v), a, varu, m['params'], db_[-1], ivar=ivar, fibfnc=self.ff, wallfield=wallfield)
                bstress.append(self.vf.deltaW_ext_membrane(self.ki.F(u), self.ki.Fdot(v), a, varu, m['params'], db_[-1], ivar=ivar, fibfnc=self.ff, stress=True, wallfield=wallfield))

            mi+=1

        return w, db_, bstress


    # set body forces (technically, no "boundary" conditions, since acting on a volume element... but implemented here for convenience)
    def bodyforce(self, bcdict, V, V_real, funcs_to_update=None):

        w = ufl.as_ufl(0)

        for b in bcdict:

            func, func_dir = fem.Function(V_real), fem.Function(V)

            # direction needs to be set
            driection = expression.template_vector()
            dir_x, dir_y, dir_z = b['dir'][0], b['dir'][1], b['dir'][2]
            dir_norm = np.sqrt(dir_x**2. + dir_y**2. + dir_z**2.)
            driection.val_x, driection.val_y, driection.val_z = dir_x/dir_norm, dir_y/dir_norm, dir_z/dir_norm
            func_dir.interpolate(driection.evaluate)

            if 'curve' in b.keys():
                assert('val' not in b.keys())
                load = expression.template()
                load.val = self.ti.timecurves(b['curve'])(self.ti.t_init)
                func.interpolate(load.evaluate)
                funcs_to_update.append({func : self.ti.timecurves(b['curve'])})
            elif 'val' in b.keys():
                assert('curve' not in b.keys())
                func.vector.set(b['val'])
            else:
                raise RuntimeError("Need to have 'curve' or 'val' specified!")

            for i in range(len(b['id'])):

                dd_ = ufl.dx(domain=self.io.mesh_master, subdomain_data=self.io.mt_d_master, subdomain_id=b['id'][i], metadata={'quadrature_degree': self.quad_degree})

                w += self.vf.deltaW_ext_bodyforce(func, func_dir, dd_)

        return w



class boundary_cond_solid(boundary_cond):

    # set Neumann BCs
    def neumann_bcs(self, bcdict, V, V_real, u, funcs_to_update=None, funcs_to_update_vec=None):

        w = ufl.as_ufl(0)

        for n in bcdict:

            try: bdim_r = n['bdim_reduction']
            except: bdim_r = 1

            if bdim_r==1: mdata = self.io.mt_b1_master
            if bdim_r==2: mdata = self.io.mt_b2_master
            if bdim_r==3: mdata = self.io.mt_b3_master

            if n['dir'] == 'xyz_ref': # reference xyz

                func = fem.Function(V)

                if 'curve' in n.keys():
                    assert('val' not in n.keys())
                    load = expression.template_vector()
                    load.val_x, load.val_y, load.val_z = self.ti.timecurves(n['curve'][0])(self.ti.t_init), self.ti.timecurves(n['curve'][1])(self.ti.t_init), self.ti.timecurves(n['curve'][2])(self.ti.t_init)
                    func.interpolate(load.evaluate)
                    funcs_to_update_vec.append({func : [self.ti.timecurves(n['curve'][0]), self.ti.timecurves(n['curve'][1]), self.ti.timecurves(n['curve'][2])]})
                elif 'val' in n.keys():
                    assert('curve' not in n.keys())
                    func.vector.set(n['val']) # currently only one value for all directions - use constant load function otherwise!
                else:
                    raise RuntimeError("Need to have 'curve' or 'val' specified!")

                for i in range(len(n['id'])):

                    db_ = ufl.ds(domain=self.io.mesh_master, subdomain_data=mdata, subdomain_id=n['id'][i], metadata={'quadrature_degree': self.quad_degree})

                    w += self.vf.deltaW_ext_neumann_ref(func, db_)

            elif n['dir'] == 'normal_ref': # reference normal

                func = fem.Function(V_real)

                if 'curve' in n.keys():
                    assert('val' not in n.keys())
                    load = expression.template()
                    load.val = self.ti.timecurves(n['curve'])(self.ti.t_init)
                    func.interpolate(load.evaluate)
                    funcs_to_update.append({func : self.ti.timecurves(n['curve'])})
                elif 'val' in n.keys():
                    assert('curve' not in n.keys())
                    func.vector.set(n['val'])
                else:
                    raise RuntimeError("Need to have 'curve' or 'val' specified!")

                for i in range(len(n['id'])):

                    db_ = ufl.ds(domain=self.io.mesh_master, subdomain_data=mdata, subdomain_id=n['id'][i], metadata={'quadrature_degree': self.quad_degree})

                    w += self.vf.deltaW_ext_neumann_normal_ref(func, db_)

            elif n['dir'] == 'xyz_cur': # current xyz

                func = fem.Function(V)

                if 'curve' in n.keys():
                    assert('val' not in n.keys())
                    load = expression.template_vector()
                    load.val_x, load.val_y, load.val_z = self.ti.timecurves(n['curve'][0])(self.ti.t_init), self.ti.timecurves(n['curve'][1])(self.ti.t_init), self.ti.timecurves(n['curve'][2])(self.ti.t_init)
                    func.interpolate(load.evaluate)
                    funcs_to_update_vec.append({func : [self.ti.timecurves(n['curve'][0]), self.ti.timecurves(n['curve'][1]), self.ti.timecurves(n['curve'][2])]})
                elif 'val' in n.keys():
                    assert('curve' not in n.keys())
                    func.vector.set(n['val']) # currently only one value for all directions - use constant load function otherwise!
                else:
                    raise RuntimeError("Need to have 'curve' or 'val' specified!")

                for i in range(len(n['id'])):

                    db_ = ufl.ds(domain=self.io.mesh_master, subdomain_data=mdata, subdomain_id=n['id'][i], metadata={'quadrature_degree': self.quad_degree})

                    w += self.vf.deltaW_ext_neumann_cur(self.ki.J(u,ext=True), self.ki.F(u,ext=True), func, db_)

            elif n['dir'] == 'normal_cur': # current normal

                func = fem.Function(V_real)

                if 'curve' in n.keys():
                    assert('val' not in n.keys())
                    load = expression.template()
                    load.val = self.ti.timecurves(n['curve'])(self.ti.t_init)
                    func.interpolate(load.evaluate)
                    funcs_to_update.append({func : self.ti.timecurves(n['curve'])})
                elif 'val' in n.keys():
                    assert('curve' not in n.keys())
                    func.vector.set(n['val'])
                else:
                    raise RuntimeError("Need to have 'curve' or 'val' specified!")

                for i in range(len(n['id'])):

                    db_ = ufl.ds(domain=self.io.mesh_master, subdomain_data=mdata, subdomain_id=n['id'][i], metadata={'quadrature_degree': self.quad_degree})

                    w += self.vf.deltaW_ext_neumann_normal_cur(self.ki.J(u,ext=True), self.ki.F(u,ext=True), func, db_)

            else:
                raise NameError("Unknown dir option for Neumann BC!")

        return w


    # set Neumann BCs for prestress
    def neumann_prestress_bcs(self, bcdict, V, V_real, funcs_to_update=None, funcs_to_update_vec=None):

        w = ufl.as_ufl(0)

        for n in bcdict:

            try: bdim_r = n['bdim_reduction']
            except: bdim_r = 1

            if bdim_r==1: mdata = self.io.mt_b1_master
            if bdim_r==2: mdata = self.io.mt_b2_master
            if bdim_r==3: mdata = self.io.mt_b3_master

            if n['dir'] == 'xyz_ref': # reference xyz

                func = fem.Function(V)

                if 'curve' in n.keys():
                    assert('val' not in n.keys())
                    load = expression.template_vector()
                    load.val_x, load.val_y, load.val_z = self.ti.timecurves(n['curve'][0])(self.ti.t_init), self.ti.timecurves(n['curve'][1])(self.ti.t_init), self.ti.timecurves(n['curve'][2])(self.ti.t_init)
                    func.interpolate(load.evaluate)
                    funcs_to_update_vec.append({func : [self.ti.timecurves(n['curve'][0]), self.ti.timecurves(n['curve'][1]), self.ti.timecurves(n['curve'][2])]})
                elif 'val' in n.keys():
                    assert('curve' not in n.keys())
                    func.vector.set(n['val']) # currently only one value for all directions - use constant load function otherwise!
                else:
                    raise RuntimeError("Need to have 'curve' or 'val' specified!")

                for i in range(len(n['id'])):

                    db_ = ufl.ds(domain=self.io.mesh_master, subdomain_data=mdata, subdomain_id=n['id'][i], metadata={'quadrature_degree': self.quad_degree})

                    w += self.vf.deltaW_ext_neumann_ref(func, db_)

            elif n['dir'] == 'normal_ref': # reference normal

                func = fem.Function(V_real)

                if 'curve' in n.keys():
                    assert('val' not in n.keys())
                    load = expression.template()
                    load.val = self.ti.timecurves(n['curve'])(self.ti.t_init)
                    func.interpolate(load.evaluate)
                    funcs_to_update.append({func : self.ti.timecurves(n['curve'])})
                elif 'val' in n.keys():
                    assert('curve' not in n.keys())
                    func.vector.set(n['val']) # currently only one value for all directions - use constant load function otherwise!
                else:
                    raise RuntimeError("Need to have 'curve' or 'val' specified!")

                for i in range(len(n['id'])):

                    db_ = ufl.ds(domain=self.io.mesh_master, subdomain_data=mdata, subdomain_id=n['id'][i], metadata={'quadrature_degree': self.quad_degree})

                    w += self.vf.deltaW_ext_neumann_normal_ref(func, db_)

            else:
                raise NameError("Unknown dir option for Neumann prestress BC!")

        return w


class boundary_cond_fluid(boundary_cond):

    # set Neumann BCs
    def neumann_bcs(self, bcdict, V, V_real, Fale=None, funcs_to_update=None, funcs_to_update_vec=None):

        w = ufl.as_ufl(0)

        for n in bcdict:

            try: bdim_r = r['bdim_reduction']
            except: bdim_r = 1

            if bdim_r==1: mdata = self.io.mt_b1_master
            if bdim_r==2: mdata = self.io.mt_b2_master
            if bdim_r==3: mdata = self.io.mt_b3_master

            if n['dir'] == 'xyz_cur': # current xyz

                func = fem.Function(V)

                if 'curve' in n.keys():
                    assert('val' not in n.keys())
                    load = expression.template_vector()
                    load.val_x, load.val_y, load.val_z = self.ti.timecurves(n['curve'][0])(self.ti.t_init), self.ti.timecurves(n['curve'][1])(self.ti.t_init), self.ti.timecurves(n['curve'][2])(self.ti.t_init)
                    func.interpolate(load.evaluate)
                    funcs_to_update_vec.append({func : [self.ti.timecurves(n['curve'][0]), self.ti.timecurves(n['curve'][1]), self.ti.timecurves(n['curve'][2])]})
                elif 'val' in n.keys():
                    assert('curve' not in n.keys())
                    func.vector.set(n['val']) # currently only one value for all directions - use constant load function otherwise!
                else:
                    raise RuntimeError("Need to have 'curve' or 'val' specified!")

                for i in range(len(n['id'])):

                    db_ = ufl.ds(domain=self.io.mesh_master, subdomain_data=mdata, subdomain_id=n['id'][i], metadata={'quadrature_degree': self.quad_degree})

                    w += self.vf.deltaW_ext_neumann_cur(func, db_, Fale=Fale)


            elif n['dir'] == 'normal_cur': # current normal

                func = fem.Function(V_real)

                if 'curve' in n.keys():
                    assert('val' not in n.keys())
                    load = expression.template()
                    load.val = self.ti.timecurves(n['curve'])(self.ti.t_init)
                    func.interpolate(load.evaluate)
                    funcs_to_update.append({func : self.ti.timecurves(n['curve'])})
                elif 'val' in n.keys():
                    assert('curve' not in n.keys())
                    func.vector.set(n['val'])
                else:
                    raise RuntimeError("Need to have 'curve' or 'val' specified!")

                for i in range(len(n['id'])):

                    db_ = ufl.ds(domain=self.io.mesh_master, subdomain_data=mdata, subdomain_id=n['id'][i], metadata={'quadrature_degree': self.quad_degree})

                    w += self.vf.deltaW_ext_neumann_normal_cur(func, db_, Fale=Fale)

            else:
                raise NameError("Unknown dir option for Neumann BC!")

        return w


    # set Neumann BCs for prestress
    def neumann_prestress_bcs(self, bcdict, V, V_real, funcs_to_update=None, funcs_to_update_vec=None):

        w = ufl.as_ufl(0)

        for n in bcdict:

            try: bdim_r = n['bdim_reduction']
            except: bdim_r = 1

            if bdim_r==1: mdata = self.io.mt_b1_master
            if bdim_r==2: mdata = self.io.mt_b2_master
            if bdim_r==3: mdata = self.io.mt_b3_master

            if n['dir'] == 'normal_ref': # reference normal - only option

                func = fem.Function(V_real)

                if 'curve' in n.keys():
                    assert('val' not in n.keys())
                    load = expression.template()
                    load.val = self.ti.timecurves(n['curve'])(self.ti.t_init)
                    func.interpolate(load.evaluate)
                    funcs_to_update.append({func : self.ti.timecurves(n['curve'])})
                elif 'val' in n.keys():
                    assert('curve' not in n.keys())
                    func.vector.set(n['val'])
                else:
                    raise RuntimeError("Need to have 'curve' or 'val' specified!")

                for i in range(len(n['id'])):

                    db_ = ufl.ds(domain=self.io.mesh_master, subdomain_data=mdata, subdomain_id=n['id'][i], metadata={'quadrature_degree': self.quad_degree})

                    w += self.vf.deltaW_ext_neumann_normal_ref(func, db_)

            else:
                raise NameError("Unknown dir option for Neumann prestress BC!")

        return w


    # set body forces (technically, no "boundary" conditions, since acting on a volume element... but implemented here for convenience)
    def bodyforce(self, bcdict, V, V_real, funcs_to_update=None, Fale=None):

        w = ufl.as_ufl(0)

        for b in bcdict:

            func, func_dir = fem.Function(V_real), fem.Function(V)

            # direction needs to be set
            driection = expression.template_vector()
            dir_x, dir_y, dir_z = b['dir'][0], b['dir'][1], b['dir'][2]
            dir_norm = np.sqrt(dir_x**2. + dir_y**2. + dir_z**2.)
            driection.val_x, driection.val_y, driection.val_z = dir_x/dir_norm, dir_y/dir_norm, dir_z/dir_norm
            func_dir.interpolate(driection.evaluate)

            if 'curve' in b.keys():
                assert('val' not in b.keys())
                load = expression.template()
                load.val = self.ti.timecurves(b['curve'])(self.ti.t_init)
                func.interpolate(load.evaluate)
                funcs_to_update.append({func : self.ti.timecurves(b['curve'])})
            elif 'val' in b.keys():
                assert('curve' not in b.keys())
                func.vector.set(b['val'])
            else:
                raise RuntimeError("Need to have 'curve' or 'val' specified!")

            for i in range(len(b['id'])):

                dd_ = ufl.dx(domain=self.io.mesh_master, subdomain_data=self.io.mt_d_master, subdomain_id=b['id'][i], metadata={'quadrature_degree': self.quad_degree})

                w += self.vf.deltaW_ext_bodyforce(func, func_dir, dd_, Fale=Fale)

        return w


    # set stabilized Neumann BCs
    def stabilized_neumann_bcs(self, bcdict, v, wel=None, Fale=None):

        w = ufl.as_ufl(0)

        for sn in bcdict:

            try: bdim_r = r['bdim_reduction']
            except: bdim_r = 1

            if bdim_r==1: mdata = self.io.mt_b1_master
            if bdim_r==2: mdata = self.io.mt_b2_master
            if bdim_r==3: mdata = self.io.mt_b3_master

            for i in range(len(sn['id'])):

                db_ = ufl.ds(domain=self.io.mesh_master, subdomain_data=mdata, subdomain_id=sn['id'][i], metadata={'quadrature_degree': self.quad_degree})

                par1 = sn['par1']
                try: par2 = sn['par2']
                except: par2 = 0.

                w += self.vf.deltaW_ext_stabilized_neumann_cur(v, par1, par2, db_, w=wel, Fale=Fale)

        return w


    # set Robin BCs
    def robin_bcs(self, bcdict, v, Fale=None):

        w = ufl.as_ufl(0)

        for r in bcdict:

            try: bdim_r = r['bdim_reduction']
            except: bdim_r = 1

            if bdim_r==1: mdata = self.io.mt_b1_master
            if bdim_r==2: mdata = self.io.mt_b2_master
            if bdim_r==3: mdata = self.io.mt_b3_master

            if r['type'] == 'dashpot':

                if r['dir'] == 'xyz_cur': # current xyz

                    for i in range(len(r['id'])):

                        db_ = ufl.ds(domain=self.io.mesh_master, subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})

                        w += self.vf.deltaW_ext_robin_dashpot(v, r['visc'], db_, Fale=Fale)

                elif r['dir'] == 'normal_cur': # current normal

                    for i in range(len(r['id'])):

                        db_ = ufl.ds(domain=self.io.mesh_master, subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})

                        w += self.vf.deltaW_ext_robin_dashpot_normal_cur(v, r['visc'], db_, Fale=Fale)

                elif r['dir'] == 'normal_cross': # cross normal

                    for i in range(len(r['id'])):

                        db_ = ufl.ds(domain=self.io.mesh_master, subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})

                        w += self.vf.deltaW_ext_robin_dashpot_normal_cross(v, r['visc'], db_, Fale=Fale)

                else:
                    raise NameError("Unknown dir option for Robin BC!")


            else:
                raise NameError("Unknown type option for Robin BC!")

        return w


    # set Robin valve BCs
    def robin_valve_bcs(self, bcdict, v, V_real, beta_, wel=None, Fale=None):

        w = ufl.as_ufl(0)

        if wel is None:
            wel_ = ufl.constantvalue.zero(self.io.mesh.topology.dim)
        else:
            wel_ = wel

        for r in bcdict:

            try: bdim_r = r['bdim_reduction']
            except: bdim_r = 1

            if bdim_r==1: mdata = self.io.mt_b1_master
            if bdim_r==2: mdata = self.io.mt_b2_master
            if bdim_r==3: mdata = self.io.mt_b3_master

            beta_.append( fem.Function(V_real) )

            for i in range(len(r['id'])):

                db_ = ufl.dS(domain=self.io.mesh_master, subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})

                w += self.vf.deltaW_ext_robin_valve(v, beta_[-1], db_, fcts='+', w=wel_, Fale=Fale)

        return w


    # set flux monitor conditions
    def flux_monitor_bcs(self, bcdict, v, qdict_, wel=None, Fale=None):

        if wel is None:
            wel_ = ufl.constantvalue.zero(self.io.mesh.topology.dim)
        else:
            wel_ = wel

        for r in bcdict:

            q = ufl.as_ufl(0)

            try: bdim_r = r['bdim_reduction']
            except: bdim_r = 1

            try: internal = r['internal']
            except: internal = False

            try: on_subdomain = r['on_subdomain']
            except: on_subdomain = False

            if internal:
                assert(not on_subdomain)
                try: fcts = r['facet_side']
                except: fcts = '+'
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

                q += self.vf.flux(v, db_, w=wel_, Fale=Fale, fcts=fcts)

            if on_subdomain:
                # entity map child to parent
                em_u = {self.io.mesh : self.io.submshes_emap[dom_u][1]}
                qdict_.append( fem.form(q, entity_maps=em_u) )
            else:
                qdict_.append( fem.form(q) )


    # set dp monitor conditions
    def dp_monitor_bcs(self, bcdict, a_u_, a_d_, pint_u_, pint_d_, pdict, wel=None, Fale=None):

        if wel is None:
            wel_ = ufl.constantvalue.zero(self.io.mesh.topology.dim)
        else:
            wel_ = wel

        # area map for integration
        if Fale is not None:
            J = ufl.det(Fale)
            ja = J*ufl.sqrt(ufl.dot(self.vf.n0, (ufl.inv(Fale)*ufl.inv(Fale).T)*self.vf.n0))
        else:
            ja = 1.0

        for r in bcdict:

            try: bdim_r = r['bdim_reduction']
            except: bdim_r = 1

            if bdim_r==1: mdata = self.io.mt_b1_master
            if bdim_r==2: mdata = self.io.mt_b2_master
            if bdim_r==3: mdata = self.io.mt_b3_master

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
            em_u = {self.io.mesh : self.io.submshes_emap[dom_u][1]}
            em_d = {self.io.mesh : self.io.submshes_emap[dom_d][1]}

            a_u_.append( fem.form(a_u, entity_maps=em_u) )
            a_d_.append( fem.form(a_d, entity_maps=em_d) )

            pint_u_.append( fem.form(pint_u, entity_maps=em_u) )
            pint_d_.append( fem.form(pint_d, entity_maps=em_d) )
