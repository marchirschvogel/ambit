#!/usr/bin/env python3

# Copyright (c) 2019-2021, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from dolfinx import DirichletBC, Function
from dolfinx.fem import locate_dofs_topological
from dolfinx.mesh import locate_entities_boundary
from ufl import ds, as_ufl

import expression


class boundary_cond():
    
    def __init__(self, bc_dict, fem_params, io, ki, vf, ti):
        
        self.bc_dict = bc_dict
        self.io = io
        self.ki = ki
        self.vf = vf
        self.ti = ti
        
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
            
            func, func_old = Function(V), Function(V)
            
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
                    self.dbcs.append( DirichletBC(func, locate_dofs_topological(V, self.io.mesh.topology.dim-bdim_r, mdata.indices[mdata.values == d['id'][i]])) )
            
            elif d['dir'] == 'x':
                for i in range(len(d['id'])):
                    self.dbcs.append( DirichletBC(func, locate_dofs_topological((V.sub(0), V.sub(0).collapse()), self.io.mesh.topology.dim-bdim_r, mdata.indices[mdata.values == d['id'][i]]), V.sub(0)) )

            elif d['dir'] == 'y':
                for i in range(len(d['id'])):
                    self.dbcs.append( DirichletBC(func, locate_dofs_topological((V.sub(1), V.sub(1).collapse()), self.io.mesh.topology.dim-bdim_r, mdata.indices[mdata.values == d['id'][i]]), V.sub(1)) )

            elif d['dir'] == 'z':
                for i in range(len(d['id'])):
                    self.dbcs.append( DirichletBC(func, locate_dofs_topological((V.sub(2), V.sub(2).collapse()), self.io.mesh.topology.dim-bdim_r, mdata.indices[mdata.values == d['id'][i]]), V.sub(2)) )

            elif d['dir'] == '2dimX':
                self.dbcs.append( DirichletBC(func, locate_dofs_topological((V.sub(0), V.sub(0).collapse()), self.io.mesh.topology.dim-bdim_r, locate_entities_boundary(self.io.mesh, self.io.mesh.topology.dim-bdim_r, self.twodimX)), V.sub(0)) )

            elif d['dir'] == '2dimY':
                self.dbcs.append( DirichletBC(func, locate_dofs_topological((V.sub(1), V.sub(1).collapse()), self.io.mesh.topology.dim-bdim_r, locate_entities_boundary(self.io.mesh, self.io.mesh.topology.dim-bdim_r, self.twodimY)), V.sub(1)) )
                
            elif d['dir'] == '2dimZ':
                self.dbcs.append( DirichletBC(func, locate_dofs_topological((V.sub(2), V.sub(2).collapse()), self.io.mesh.topology.dim-bdim_r, locate_entities_boundary(self.io.mesh, self.io.mesh.topology.dim-bdim_r, self.twodimZ)), V.sub(2)) )

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


class boundary_cond_solid(boundary_cond):

    # set Neumann BCs
    def neumann_bcs(self, V, V_real, u, u_old):
        
        w, w_old = as_ufl(0), as_ufl(0)
        
        for n in self.bc_dict['neumann']:
            
            try: bdim_r = n['bdim_reduction']
            except: bdim_r = 1
            
            if bdim_r==1: mdata = self.io.mt_b1
            if bdim_r==2: mdata = self.io.mt_b2
            if bdim_r==3: mdata = self.io.mt_b3
            
            if n['type'] == 'pk1':
                
                if n['dir'] == 'xyz':
                
                    func, func_old = Function(V), Function(V)
                    
                    if 'curve' in n.keys():
                        load = expression.template_vector()
                        load.val_x, load.val_y, load.val_z = self.ti.timecurves(n['curve'][0])(self.ti.t_init), self.ti.timecurves(n['curve'][1])(self.ti.t_init), self.ti.timecurves(n['curve'][2])(self.ti.t_init)
                        func.interpolate(load.evaluate), func_old.interpolate(load.evaluate)
                        self.ti.funcs_to_update_vec.append({func : [self.ti.timecurves(n['curve'][0]), self.ti.timecurves(n['curve'][1]), self.ti.timecurves(n['curve'][2])]})
                        self.ti.funcs_to_update_vec_old.append({func_old : [self.ti.timecurves(n['curve'][0]), self.ti.timecurves(n['curve'][1]), self.ti.timecurves(n['curve'][2])]})
                    else:
                        func.vector.set(n['val']) # currently only one value for all directions - use constant load function otherwise!
                        func_old.vector.set(n['val'])
                    
                    for i in range(len(n['id'])):
                        
                        db_ = ds(subdomain_data=mdata, subdomain_id=n['id'][i], metadata={'quadrature_degree': self.quad_degree})
                    
                        w     += self.vf.deltaW_ext_neumann_ref(func, db_)
                        w_old += self.vf.deltaW_ext_neumann_ref(func_old, db_)
                    
                elif n['dir'] == 'normal': # reference normal
                    
                    func, func_old = Function(V_real), Function(V_real)
                    
                    if 'curve' in n.keys():
                        load = expression.template()
                        load.val = self.ti.timecurves(n['curve'])(self.ti.t_init)
                        func.interpolate(load.evaluate), func_old.interpolate(load.evaluate)
                        self.ti.funcs_to_update.append({func : self.ti.timecurves(n['curve'])})
                        self.ti.funcs_to_update_old.append({func_old : self.ti.timecurves(n['curve'])})
                    else:
                        func.vector.set(n['val'])
                        func_old.vector.set(n['val'])
                    
                    for i in range(len(n['id'])):
                        
                        db_ = ds(subdomain_data=mdata, subdomain_id=n['id'][i], metadata={'quadrature_degree': self.quad_degree})
                    
                        w     += self.vf.deltaW_ext_neumann_refnormal(func, db_)
                        w_old += self.vf.deltaW_ext_neumann_refnormal(func_old, db_)
                    
                else:
                    raise NameError("Unknown dir option for Neumann BC!")


            elif n['type'] == 'true':
                
                if n['dir'] == 'normal': # true normal
                    
                    func, func_old = Function(V_real), Function(V_real)
                    
                    if 'curve' in n.keys():
                        load = expression.template()
                        load.val = self.ti.timecurves(n['curve'])(self.ti.t_init)
                        func.interpolate(load.evaluate), func_old.interpolate(load.evaluate)
                        self.ti.funcs_to_update.append({func : self.ti.timecurves(n['curve'])})
                        self.ti.funcs_to_update_old.append({func_old : self.ti.timecurves(n['curve'])})
                    else:
                        func.vector.set(n['val'])
                        func_old.vector.set(n['val'])

                    for i in range(len(n['id'])):
                        
                        db_ = ds(subdomain_data=mdata, subdomain_id=n['id'][i], metadata={'quadrature_degree': self.quad_degree})

                        w     += self.vf.deltaW_ext_neumann_true(self.ki.J(u), self.ki.F(u), func, db_)
                        w_old += self.vf.deltaW_ext_neumann_true(self.ki.J(u_old), self.ki.F(u_old), func_old, db_)
                    
                else:
                    raise NameError("Unknown dir option for Neumann BC!")

            else:
                raise NameError("Unknown type option for Neumann BC!")

        return w, w_old


    # set Robin BCs
    def robin_bcs(self, u, vel, u_old, v_old, u_pre=None):
        
        w, w_old = as_ufl(0), as_ufl(0)
        
        for r in self.bc_dict['robin']:
            
            try: bdim_r = r['bdim_reduction']
            except: bdim_r = 1

            if bdim_r==1: mdata = self.io.mt_b1
            if bdim_r==2: mdata = self.io.mt_b2
            if bdim_r==3: mdata = self.io.mt_b3

            if r['type'] == 'spring':
                
                if r['dir'] == 'xyz':
                    
                    for i in range(len(r['id'])):
                    
                        db_ = ds(subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})
                        
                        w     += self.vf.deltaW_ext_robin_spring(u, r['stiff'], db_, u_pre)
                        w_old += self.vf.deltaW_ext_robin_spring(u_old, r['stiff'], db_, u_pre)
                    
                    
                elif r['dir'] == 'normal': # reference normal
                    
                    for i in range(len(r['id'])):
                        
                        db_ = ds(subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})
                
                        w     += self.vf.deltaW_ext_robin_spring_normal(u, r['stiff'], db_, u_pre)
                        w_old += self.vf.deltaW_ext_robin_spring_normal(u_old, r['stiff'], db_, u_pre) 

                else:
                    raise NameError("Unknown dir option for Robin BC!")


            elif r['type'] == 'dashpot':
                
                if r['dir'] == 'xyz':
                    
                    for i in range(len(r['id'])):
                        
                        db_ = ds(subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})
                    
                        w     += self.vf.deltaW_ext_robin_dashpot(vel, r['visc'], db_)
                        w_old += self.vf.deltaW_ext_robin_dashpot(v_old, r['visc'], db_) 
                    

                elif r['dir'] == 'normal': # reference normal
                    
                    for i in range(len(r['id'])):
                        
                        db_ = ds(subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})
                
                        w     += self.vf.deltaW_ext_robin_dashpot_normal(vel, r['visc'], db_)
                        w_old += self.vf.deltaW_ext_robin_dashpot_normal(v_old, r['visc'], db_) 

                else:
                    raise NameError("Unknown dir option for Robin BC!")


            else:
                raise NameError("Unknown type option for Robin BC!")
            
        return w, w_old


    # set membrane surface BCs
    def membranesurf_bcs(self, u, u_old):
        
        w, w_old = as_ufl(0), as_ufl(0)
        
        for m in self.bc_dict['membrane']:
            
            try: bdim_r = m['bdim_reduction']
            except: bdim_r = 1

            if bdim_r==1: mdata = self.io.mt_b1
            if bdim_r==2: mdata = self.io.mt_b2
            if bdim_r==3: mdata = self.io.mt_b3
                    
            for i in range(len(m['id'])):
            
                db_ = ds(subdomain_data=mdata, subdomain_id=m['id'][i], metadata={'quadrature_degree': self.quad_degree})
                
                w     += self.vf.deltaW_ext_membrane(self.ki.F(u), m['params'], db_)
                w_old += self.vf.deltaW_ext_membrane(self.ki.F(u_old), m['params'], db_)

        return w, w_old
    





class boundary_cond_fluid(boundary_cond):

    # set Neumann BCs
    def neumann_bcs(self, V, V_real):
        
        w, w_old = as_ufl(0), as_ufl(0)
        
        for n in self.bc_dict['neumann']:
            
            try: bdim_r = r['bdim_reduction']
            except: bdim_r = 1

            if bdim_r==1: mdata = self.io.mt_b1
            if bdim_r==2: mdata = self.io.mt_b2
            if bdim_r==3: mdata = self.io.mt_b3

            if n['dir'] == 'xyz':
            
                func, func_old = Function(V), Function(V)
                
                if 'curve' in n.keys():
                    load = expression.template_vector()
                    load.val_x, load.val_y, load.val_z = self.ti.timecurves(n['curve'][0])(self.ti.t_init), self.ti.timecurves(n['curve'][1])(self.ti.t_init), self.ti.timecurves(n['curve'][2])(self.ti.t_init)
                    func.interpolate(load.evaluate), func_old.interpolate(load.evaluate)
                    self.ti.funcs_to_update_vec.append({func : [self.ti.timecurves(n['curve'][0]), self.ti.timecurves(n['curve'][1]), self.ti.timecurves(n['curve'][2])]})
                    self.ti.funcs_to_update_vec_old.append({func_old : [self.ti.timecurves(n['curve'][0]), self.ti.timecurves(n['curve'][1]), self.ti.timecurves(n['curve'][2])]})
                else:
                    func.vector.set(n['val']) # currently only one value for all directions - use constant load function otherwise!
                    func_old.vector.set(n['val'])
                
                for i in range(len(n['id'])):
                    
                    db_ = ds(subdomain_data=mdata, subdomain_id=n['id'][i], metadata={'quadrature_degree': self.quad_degree})
                
                    w     += self.vf.deltaP_ext_neumann(func, db_)
                    w_old += self.vf.deltaP_ext_neumann(func_old, db_)
                
                
            elif n['dir'] == 'normal': # reference normal
                
                func, func_old = Function(V_real), Function(V_real)
                
                if 'curve' in n.keys():
                    load = expression.template()
                    load.val = self.ti.timecurves(n['curve'])(self.ti.t_init)
                    func.interpolate(load.evaluate), func_old.interpolate(load.evaluate)
                    self.ti.funcs_to_update.append({func : self.ti.timecurves(n['curve'])})
                    self.ti.funcs_to_update_old.append({func_old : self.ti.timecurves(n['curve'])})
                else:
                    func.vector.set(n['val'])
                    func_old.vector.set(n['val'])
                
                for i in range(len(n['id'])):
                    
                    db_ = ds(subdomain_data=mdata, subdomain_id=n['id'][i], metadata={'quadrature_degree': self.quad_degree})
                
                    w     += self.vf.deltaP_ext_neumann_normal(func, db_)
                    w_old += self.vf.deltaP_ext_neumann_normal(func_old, db_)
                
            else:
                raise NameError("Unknown dir option for Neumann BC!")

        return w, w_old


    # set Robin BCs
    def robin_bcs(self, v, v_old):
        
        w, w_old = as_ufl(0), as_ufl(0)
        
        for r in self.bc_dict['robin']:
            
            try: bdim_r = r['bdim_reduction']
            except: bdim_r = 1

            if bdim_r==1: mdata = self.io.mt_b1
            if bdim_r==2: mdata = self.io.mt_b2
            if bdim_r==3: mdata = self.io.mt_b3

            if r['type'] == 'dashpot':
                
                if r['dir'] == 'xyz':
                    
                    for i in range(len(r['id'])):
                        
                        db_ = ds(subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})
                    
                        w     += self.vf.deltaP_ext_robin_dashpot(v, r['visc'], db_)
                        w_old += self.vf.deltaP_ext_robin_dashpot(v_old, r['visc'], db_) 
                    
                elif r['dir'] == 'normal': # reference normal
                
                    for i in range(len(r['id'])):
                        
                        db_ = ds(subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})
                
                        w     += self.vf.deltaP_ext_robin_dashpot_normal(v, r['visc'], db_)
                        w_old += self.vf.deltaP_ext_robin_dashpot_normal(v_old, r['visc'], db_) 

                else:
                    raise NameError("Unknown dir option for Robin BC!")


            else:
                raise NameError("Unknown type option for Robin BC!")
            
        return w, w_old
