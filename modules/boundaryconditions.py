#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from dolfinx import fem, mesh
import ufl

import expression


class boundary_cond():
    
    def __init__(self, bc_dict, fem_params, io, vf, ti, ki=None):
        
        self.bc_dict = bc_dict
        self.io = io
        self.vf = vf
        self.ti = ti
        self.ki = ki
        
        self.quad_degree = fem_params['quad_degree']
        
        self.dbcs = []

    
    # set Dirichlet BCs (should probably be overloaded for problems that do not have vector variables...)
    def dirichlet_bcs(self, V):
        
        for d in self.bc_dict['dirichlet']:
            
            try: bdim_r = d['bdim_reduction']
            except: bdim_r = 1
            
            if bdim_r==1: mdata = self.io.mt_b1
            if bdim_r==2: mdata = self.io.mt_b2
            if bdim_r==3: mdata = self.io.mt_b3
            
            func, func_old = fem.Function(V), fem.Function(V)
            
            if 'curve' in d.keys():
                load = expression.template_vector()
                if d['dir'] == 'all': curve_x, curve_y, curve_z = d['curve'][0], d['curve'][1], d['curve'][2]
                else:                 curve_x, curve_y, curve_z = d['curve'], d['curve'], d['curve']
                load.val_x, load.val_y, load.val_z = self.ti.timecurves(curve_x)(self.ti.t_init), self.ti.timecurves(curve_y)(self.ti.t_init), self.ti.timecurves(curve_z)(self.ti.t_init)
                func.interpolate(load.evaluate), func_old.interpolate(load.evaluate)
                self.ti.funcs_to_update_vec.append({func : [self.ti.timecurves(curve_x), self.ti.timecurves(curve_y), self.ti.timecurves(curve_z)]})
                self.ti.funcs_to_update_vec_old.append({func_old : [self.ti.timecurves(curve_x), self.ti.timecurves(curve_y), self.ti.timecurves(curve_z)]})
            else:
                func.vector.set(d['val'])
                func_old.vector.set(d['val'])

            if d['dir'] == 'all':
                for i in range(len(d['id'])):
                    self.dbcs.append( fem.dirichletbc(func, fem.locate_dofs_topological(V, self.io.mesh.topology.dim-bdim_r, mdata.indices[mdata.values == d['id'][i]])) )
            
            elif d['dir'] == 'x':
                for i in range(len(d['id'])):
                    self.dbcs.append( fem.dirichletbc(func.sub(0), fem.locate_dofs_topological(V.sub(0), self.io.mesh.topology.dim-bdim_r, mdata.indices[mdata.values == d['id'][i]])) )

            elif d['dir'] == 'y':
                for i in range(len(d['id'])):
                    self.dbcs.append( fem.dirichletbc(func.sub(1), fem.locate_dofs_topological(V.sub(1), self.io.mesh.topology.dim-bdim_r, mdata.indices[mdata.values == d['id'][i]])) )

            elif d['dir'] == 'z':
                for i in range(len(d['id'])):
                    self.dbcs.append( fem.dirichletbc(func.sub(2), fem.locate_dofs_topological(V.sub(2), self.io.mesh.topology.dim-bdim_r, mdata.indices[mdata.values == d['id'][i]])) )

            elif d['dir'] == '2dimX':
                self.dbcs.append( fem.dirichletbc(func.sub(0), fem.locate_dofs_topological(V.sub(0), self.io.mesh.topology.dim-bdim_r, mesh.locate_entities_boundary(self.io.mesh, self.io.mesh.topology.dim-bdim_r, self.twodimX))) )

            elif d['dir'] == '2dimY':
                self.dbcs.append( fem.dirichletbc(func.sub(1), fem.locate_dofs_topological(V.sub(1), self.io.mesh.topology.dim-bdim_r, mesh.locate_entities_boundary(self.io.mesh, self.io.mesh.topology.dim-bdim_r, self.twodimY))) )

            elif d['dir'] == '2dimZ':
                self.dbcs.append( fem.dirichletbc(func.sub(2), fem.locate_dofs_topological(V.sub(2), self.io.mesh.topology.dim-bdim_r, mesh.locate_entities_boundary(self.io.mesh, self.io.mesh.topology.dim-bdim_r, self.twodimZ))) )

            else:
                raise NameError("Unknown dir option for Dirichlet BC!")

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
    def robin_bcs(self, u, v, u_pre=None):
        
        w = ufl.as_ufl(0)
        
        for r in self.bc_dict['robin']:
            
            try: bdim_r = r['bdim_reduction']
            except: bdim_r = 1

            if bdim_r==1: mdata = self.io.mt_b1
            if bdim_r==2: mdata = self.io.mt_b2
            if bdim_r==3: mdata = self.io.mt_b3

            if r['type'] == 'spring':
                
                if r['dir'] == 'xyz':
                    
                    for i in range(len(r['id'])):
                    
                        db_ = ufl.ds(subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})
                        
                        w     += self.vf.deltaW_ext_robin_spring(u, r['stiff'], db_, u_pre)

                elif r['dir'] == 'normal': # reference normal
                    
                    for i in range(len(r['id'])):
                        
                        db_ = ufl.ds(subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})
                
                        w     += self.vf.deltaW_ext_robin_spring_normal(u, r['stiff'], db_, u_pre)

                else:
                    raise NameError("Unknown dir option for Robin BC!")


            elif r['type'] == 'dashpot':
                
                if r['dir'] == 'xyz':
                    
                    for i in range(len(r['id'])):
                        
                        db_ = ufl.ds(subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})
                    
                        w     += self.vf.deltaW_ext_robin_dashpot(v, r['visc'], db_)

                elif r['dir'] == 'normal': # reference normal
                    
                    for i in range(len(r['id'])):
                        
                        db_ = ufl.ds(subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})
                
                        w     += self.vf.deltaW_ext_robin_dashpot_normal(v, r['visc'], db_)

                else:
                    raise NameError("Unknown dir option for Robin BC!")


            else:
                raise NameError("Unknown type option for Robin BC!")
            
        return w


    # set weak Dirichlet BCs
    def weak_dirichlet_bcs(self, u, uD, var_stress):

        w = ufl.as_ufl(0)
        
        for n in self.bc_dict['dirichlet_weak']:

            try: bdim_r = n['bdim_reduction']
            except: bdim_r = 1
            
            if bdim_r==1: mdata = self.io.mt_b1
            if bdim_r==2: mdata = self.io.mt_b2
            if bdim_r==3: mdata = self.io.mt_b3
            
            for i in range(len(n['id'])):
                
                db_ = ufl.ds(subdomain_data=mdata, subdomain_id=n['id'][i], metadata={'quadrature_degree': self.quad_degree})
                
                beta = n['beta']
                
                w += self.vf.deltaW_int_nitsche_dirichlet(u, uD, var_stress, beta, db_)

        return w


    # set membrane surface BCs
    def membranesurf_bcs(self, u, v, a):
        
        w = ufl.as_ufl(0)
        
        for m in self.bc_dict['membrane']:
            
            try: bdim_r = m['bdim_reduction']
            except: bdim_r = 1

            if bdim_r==1: mdata = self.io.mt_b1
            if bdim_r==2: mdata = self.io.mt_b2
            if bdim_r==3: mdata = self.io.mt_b3
                    
            for i in range(len(m['id'])):
            
                db_ = ufl.ds(subdomain_data=mdata, subdomain_id=m['id'][i], metadata={'quadrature_degree': self.quad_degree})
                
                w     += self.vf.deltaW_ext_membrane(self.ki.F(u), self.ki.Fdot(v), a, m['params'], db_)

        return w


class boundary_cond_solid(boundary_cond):

    # set Neumann BCs
    def neumann_bcs(self, V, V_real, u, funcs_to_update=None, funcs_to_update_vec=None):
        
        w = ufl.as_ufl(0)
        
        for n in self.bc_dict['neumann']:
            
            try: bdim_r = n['bdim_reduction']
            except: bdim_r = 1
            
            if bdim_r==1: mdata = self.io.mt_b1
            if bdim_r==2: mdata = self.io.mt_b2
            if bdim_r==3: mdata = self.io.mt_b3
            
            if n['type'] == 'pk1':
                
                if n['dir'] == 'xyz':
                
                    func = fem.Function(V)
                    
                    if 'curve' in n.keys():
                        load = expression.template_vector()
                        load.val_x, load.val_y, load.val_z = self.ti.timecurves(n['curve'][0])(self.ti.t_init), self.ti.timecurves(n['curve'][1])(self.ti.t_init), self.ti.timecurves(n['curve'][2])(self.ti.t_init)
                        func.interpolate(load.evaluate)
                        funcs_to_update_vec.append({func : [self.ti.timecurves(n['curve'][0]), self.ti.timecurves(n['curve'][1]), self.ti.timecurves(n['curve'][2])]})
                    else:
                        func.vector.set(n['val']) # currently only one value for all directions - use constant load function otherwise!
                    
                    for i in range(len(n['id'])):
                        
                        db_ = ufl.ds(subdomain_data=mdata, subdomain_id=n['id'][i], metadata={'quadrature_degree': self.quad_degree})
                    
                        w += self.vf.deltaW_ext_neumann_ref(func, db_)
                    
                elif n['dir'] == 'normal': # reference normal
                    
                    func = fem.Function(V_real)
                    
                    if 'curve' in n.keys():
                        load = expression.template()
                        load.val = self.ti.timecurves(n['curve'])(self.ti.t_init)
                        func.interpolate(load.evaluate)
                        funcs_to_update.append({func : self.ti.timecurves(n['curve'])})
                    else:
                        func.vector.set(n['val'])
                    
                    for i in range(len(n['id'])):
                        
                        db_ = ufl.ds(subdomain_data=mdata, subdomain_id=n['id'][i], metadata={'quadrature_degree': self.quad_degree})
                    
                        w += self.vf.deltaW_ext_neumann_refnormal(func, db_)
                    
                else:
                    raise NameError("Unknown dir option for Neumann BC!")


            elif n['type'] == 'true':
                
                if n['dir'] == 'normal': # true normal
                    
                    func = fem.Function(V_real)
                    
                    if 'curve' in n.keys():
                        load = expression.template()
                        load.val = self.ti.timecurves(n['curve'])(self.ti.t_init)
                        func.interpolate(load.evaluate)
                        funcs_to_update.append({func : self.ti.timecurves(n['curve'])})
                    else:
                        func.vector.set(n['val'])

                    for i in range(len(n['id'])):
                        
                        db_ = ufl.ds(subdomain_data=mdata, subdomain_id=n['id'][i], metadata={'quadrature_degree': self.quad_degree})

                        w += self.vf.deltaW_ext_neumann_true(self.ki.J(u,ext=True), self.ki.F(u,ext=True), func, db_)
                    
                else:
                    raise NameError("Unknown dir option for Neumann BC!")

            else:
                raise NameError("Unknown type option for Neumann BC!")

        return w


class boundary_cond_fluid(boundary_cond):

    # set Neumann BCs
    def neumann_bcs(self, V, V_real, Fale=None, funcs_to_update=None, funcs_to_update_vec=None):
        
        w = ufl.as_ufl(0)
        
        for n in self.bc_dict['neumann']:
            
            try: bdim_r = r['bdim_reduction']
            except: bdim_r = 1

            if bdim_r==1: mdata = self.io.mt_b1
            if bdim_r==2: mdata = self.io.mt_b2
            if bdim_r==3: mdata = self.io.mt_b3

            if n['dir'] == 'xyz':
            
                func = fem.Function(V)
                
                if 'curve' in n.keys():
                    load = expression.template_vector()
                    load.val_x, load.val_y, load.val_z = self.ti.timecurves(n['curve'][0])(self.ti.t_init), self.ti.timecurves(n['curve'][1])(self.ti.t_init), self.ti.timecurves(n['curve'][2])(self.ti.t_init)
                    func.interpolate(load.evaluate)
                    funcs_to_update_vec.append({func : [self.ti.timecurves(n['curve'][0]), self.ti.timecurves(n['curve'][1]), self.ti.timecurves(n['curve'][2])]})
                else:
                    func.vector.set(n['val']) # currently only one value for all directions - use constant load function otherwise!
                
                for i in range(len(n['id'])):
                    
                    db_ = ufl.ds(subdomain_data=mdata, subdomain_id=n['id'][i], metadata={'quadrature_degree': self.quad_degree})
                
                    w += self.vf.deltaW_ext_neumann(func, db_)
                
            elif n['dir'] == 'normal': # reference normal
                
                func = fem.Function(V_real)
                
                if 'curve' in n.keys():
                    load = expression.template()
                    load.val = self.ti.timecurves(n['curve'])(self.ti.t_init)
                    func.interpolate(load.evaluate)
                    funcs_to_update.append({func : self.ti.timecurves(n['curve'])})
                else:
                    func.vector.set(n['val'])
                
                for i in range(len(n['id'])):
                    
                    db_ = ufl.ds(subdomain_data=mdata, subdomain_id=n['id'][i], metadata={'quadrature_degree': self.quad_degree})
                
                    w += self.vf.deltaW_ext_neumann_normal(func, db_, Fale=Fale)
                
            else:
                raise NameError("Unknown dir option for Neumann BC!")

        return w


class boundary_cond_ale(boundary_cond):

    # set Neumann BCs
    def neumann_bcs(self, V, V_real, funcs_to_update=None, funcs_to_update_vec=None):
        
        w = ufl.as_ufl(0)
        
        for n in self.bc_dict['neumann']:
            
            try: bdim_r = r['bdim_reduction']
            except: bdim_r = 1

            if bdim_r==1: mdata = self.io.mt_b1
            if bdim_r==2: mdata = self.io.mt_b2
            if bdim_r==3: mdata = self.io.mt_b3
            
            if n['type'] == 'pk1':

                if n['dir'] == 'xyz':
                
                    func = fem.Function(V)
                    
                    if 'curve' in n.keys():
                        load = expression.template_vector()
                        load.val_x, load.val_y, load.val_z = self.ti.timecurves(n['curve'][0])(self.ti.t_init), self.ti.timecurves(n['curve'][1])(self.ti.t_init), self.ti.timecurves(n['curve'][2])(self.ti.t_init)
                        func.interpolate(load.evaluate)
                        funcs_to_update_vec.append({func : [self.ti.timecurves(n['curve'][0]), self.ti.timecurves(n['curve'][1]), self.ti.timecurves(n['curve'][2])]})
                    else:
                        func.vector.set(n['val']) # currently only one value for all directions - use constant load function otherwise!
                    
                    for i in range(len(n['id'])):
                        
                        db_ = ufl.ds(subdomain_data=mdata, subdomain_id=n['id'][i], metadata={'quadrature_degree': self.quad_degree})
                    
                        w += self.vf.deltaW_ext_neumann_ref(func, db_)
                    
                else:
                    raise NameError("Unknown dir option for Neumann BC!")

            else:
                raise NameError("Unknown type option for Neumann BC!")

        return w
