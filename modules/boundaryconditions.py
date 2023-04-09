#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from dolfinx import fem, mesh
import ufl
from petsc4py import PETSc

import expression


class boundary_cond():
    
    def __init__(self, fem_params, io, vf, ti, ki=None, ff=None):

        self.io = io
        self.vf = vf
        self.ti = ti
        self.ki = ki
        self.ff = ff
        
        self.quad_degree = fem_params['quad_degree']
        
        self.dbcs = []

    
    # set Dirichlet BCs (should probably be overloaded for problems that do not have vector variables...)
    def dirichlet_bcs(self, bcdict, V):
        
        for d in bcdict:
            
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

            if bdim_r==1: mdata = self.io.mt_b1
            if bdim_r==2: mdata = self.io.mt_b2
            if bdim_r==3: mdata = self.io.mt_b3

            if r['type'] == 'spring':
                
                if r['dir'] == 'xyz_ref': # reference xyz
                    
                    for i in range(len(r['id'])):
                    
                        db_ = ufl.ds(subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})
                        
                        w += self.vf.deltaW_ext_robin_spring(u, r['stiff'], db_, u_pre)

                elif r['dir'] == 'normal_ref': # reference normal
                    
                    for i in range(len(r['id'])):
                        
                        db_ = ufl.ds(subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})
                
                        w += self.vf.deltaW_ext_robin_spring_normal_ref(u, r['stiff'], db_, u_pre)

                else:
                    raise NameError("Unknown dir option for Robin BC!")


            elif r['type'] == 'dashpot':
                
                if r['dir'] == 'xyz_ref':
                    
                    for i in range(len(r['id'])):
                        
                        db_ = ufl.ds(subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})
                    
                        w     += self.vf.deltaW_ext_robin_dashpot(v, r['visc'], db_)

                elif r['dir'] == 'normal_ref': # reference normal
                    
                    for i in range(len(r['id'])):
                        
                        db_ = ufl.ds(subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})
                
                        w += self.vf.deltaW_ext_robin_dashpot_normal_ref(v, r['visc'], db_)

                else:
                    raise NameError("Unknown dir option for Robin BC!")


            else:
                raise NameError("Unknown type option for Robin BC!")
            
        return w


    # set membrane surface BCs
    def membranesurf_bcs(self, bcdict, u, v, a, ivar=None):
        
        w = ufl.as_ufl(0)

        for m in bcdict:
            
            try: bdim_r = m['bdim_reduction']
            except: bdim_r = 1

            if bdim_r==1: mdata = self.io.mt_b1
            if bdim_r==2: mdata = self.io.mt_b2
            if bdim_r==3: mdata = self.io.mt_b3
                    
            for i in range(len(m['id'])):
            
                db_ = ufl.ds(subdomain_data=mdata, subdomain_id=m['id'][i], metadata={'quadrature_degree': self.quad_degree})
                
                w += self.vf.deltaW_ext_membrane(self.ki.F(u), self.ki.Fdot(v), a, m['params'], db_, ivar=ivar, fibfnc=self.ff)

        return w


class boundary_cond_solid(boundary_cond):

    # set Neumann BCs
    def neumann_bcs(self, bcdict, V, V_real, u, funcs_to_update=None, funcs_to_update_vec=None):
        
        w = ufl.as_ufl(0)
        
        for n in bcdict:
            
            try: bdim_r = n['bdim_reduction']
            except: bdim_r = 1
            
            if bdim_r==1: mdata = self.io.mt_b1
            if bdim_r==2: mdata = self.io.mt_b2
            if bdim_r==3: mdata = self.io.mt_b3

            if n['dir'] == 'xyz_ref': # reference xyz
            
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
                
            elif n['dir'] == 'normal_ref': # reference normal
                
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
                
                    w += self.vf.deltaW_ext_neumann_normal_ref(func, db_)

            elif n['dir'] == 'xyz_cur': # current xyz
            
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
                
                    w += self.vf.deltaW_ext_neumann_cur(self.ki.J(u,ext=True), self.ki.F(u,ext=True), func, db_)

            elif n['dir'] == 'normal_cur': # current normal
                
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

                    w += self.vf.deltaW_ext_neumann_normal_cur(self.ki.J(u,ext=True), self.ki.F(u,ext=True), func, db_)
                
            else:
                raise NameError("Unknown dir option for Neumann BC!")

        return w


    # set Neumann BCs for prestress
    def neumann_prestress_bcs(self, bcdict, V, V_real, u, funcs_to_update=None, funcs_to_update_vec=None):
        
        w = ufl.as_ufl(0)
        
        for n in bcdict:
            
            try: bdim_r = n['bdim_reduction']
            except: bdim_r = 1
            
            if bdim_r==1: mdata = self.io.mt_b1
            if bdim_r==2: mdata = self.io.mt_b2
            if bdim_r==3: mdata = self.io.mt_b3

            if n['dir'] == 'xyz_ref': # reference xyz
            
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
                
            elif n['dir'] == 'normal_ref': # reference normal
                
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

            if bdim_r==1: mdata = self.io.mt_b1
            if bdim_r==2: mdata = self.io.mt_b2
            if bdim_r==3: mdata = self.io.mt_b3

            if n['dir'] == 'xyz_cur': # current xyz
            
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
                
                    w += self.vf.deltaW_ext_neumann_cur(func, db_, Fale=Fale)


            elif n['dir'] == 'normal_cur': # current normal
                
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
                
                    w += self.vf.deltaW_ext_neumann_normal_cur(func, db_, Fale=Fale)
                
            else:
                raise NameError("Unknown dir option for Neumann BC!")

        return w


    # set stabilized Neumann BCs
    def stabilized_neumann_bcs(self, bcdict, v, wel=None, Fale=None):
        
        w = ufl.as_ufl(0)
        
        for sn in bcdict:
            
            try: bdim_r = r['bdim_reduction']
            except: bdim_r = 1

            if bdim_r==1: mdata = self.io.mt_b1
            if bdim_r==2: mdata = self.io.mt_b2
            if bdim_r==3: mdata = self.io.mt_b3
            
            for i in range(len(sn['id'])):
                
                db_ = ufl.ds(subdomain_data=mdata, subdomain_id=sn['id'][i], metadata={'quadrature_degree': self.quad_degree})
                
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

            if bdim_r==1: mdata = self.io.mt_b1
            if bdim_r==2: mdata = self.io.mt_b2
            if bdim_r==3: mdata = self.io.mt_b3

            if r['type'] == 'dashpot':
                
                if r['dir'] == 'xyz_cur': # current xyz
                    
                    for i in range(len(r['id'])):
                        
                        db_ = ufl.ds(subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})
                    
                        w += self.vf.deltaW_ext_robin_dashpot(v, r['visc'], db_, Fale=Fale)

                elif r['dir'] == 'normal_cur': # current normal
                    
                    for i in range(len(r['id'])):
                        
                        db_ = ufl.ds(subdomain_data=mdata, subdomain_id=r['id'][i], metadata={'quadrature_degree': self.quad_degree})
                
                        w += self.vf.deltaW_ext_robin_dashpot_normal_cur(v, r['visc'], db_, Fale=Fale)

                else:
                    raise NameError("Unknown dir option for Robin BC!")


            else:
                raise NameError("Unknown type option for Robin BC!")
            
        return w
