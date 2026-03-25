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


class boundary_cond:
    def __init__(
        self,
        pb,
        dim=None,
        V_field=None,
        Vdisc_scalar=None,
    ):

        self.pb = pb

        if dim is None:
            self.dim = V_field.mesh.topology.dim
        else:
            self.dim = dim

        # continuous function space of primary field (e.g. displacement or velocity)
        self.V_field = V_field
        # discontinuous scalar function space
        self.Vdisc_scalar = Vdisc_scalar

        self.have_dirichlet_fileseries = False

    # set Dirichlet BCs (should probably be overloaded for problems that do not have vector variables...)
    def dirichlet_bcs(self, bcdict, dbcs, V_dbc=None):
        # use V_field if no extra function space is given
        if V_dbc is None:
            V_dbc = self.V_field

        for d in bcdict:
            codim = d.get("codimension", self.dim - 1)

            if codim == self.dim:
                mdata = self.pb.mt_d
            if codim == self.dim - 1:
                mdata = self.pb.mt_b
            if codim == self.dim - 2:
                mdata = self.pb.mt_sb
            if codim == self.dim - 3:
                mdata = self.pb.mt_ssb

            func = fem.Function(V_dbc)

            if "curve" in d.keys():
                assert "val" not in d.keys() and "expression" not in d.keys() and "file" not in d.keys() and "fileseries" not in d.keys()
                load = expression.template_vector(dim=self.dim)
                if d["dir"] == "all":
                    curve_x, curve_y, curve_z = (
                        d["curve"][0],
                        d["curve"][1],
                        d["curve"][2],
                    )
                else:
                    curve_x, curve_y, curve_z = (
                        d["curve"],
                        d["curve"],
                        d["curve"],
                    )
                load.val_x, load.val_y, load.val_z = (
                    self.pb.ti.timecurves(curve_x)(self.pb.ti.t_init),
                    self.pb.ti.timecurves(curve_y)(self.pb.ti.t_init),
                    self.pb.ti.timecurves(curve_z)(self.pb.ti.t_init),
                )
                func.interpolate(load.evaluate)
                func.x.petsc_vec.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT,
                    mode=PETSc.ScatterMode.FORWARD,
                )
                self.pb.ti.funcs_to_update_vec.append(
                    {
                        func: [
                            self.pb.ti.timecurves(curve_x),
                            self.pb.ti.timecurves(curve_y),
                            self.pb.ti.timecurves(curve_z),
                        ]
                    }
                )
                self.pb.ti.funcs_to_update_vec_old.append({None: -1})  # DBCs don't need an old state
            elif "val" in d.keys():
                assert "curve" not in d.keys() and "expression" not in d.keys() and "file" not in d.keys() and "fileseries" not in d.keys()
                func.x.petsc_vec.set(d["val"])
            elif "expression" in d.keys():
                assert "curve" not in d.keys() and "val" not in d.keys() and "file" not in d.keys() and "fileseries" not in d.keys()
                expr = d["expression"]()
                expr.t = self.pb.ti.t_init
                func.interpolate(expr.evaluate)
                func.x.petsc_vec.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT,
                    mode=PETSc.ScatterMode.FORWARD,
                )
                self.pb.ti.funcsexpr_to_update_vec[func] = expr
                self.pb.ti.funcsexpr_to_update_vec_old[func] = None  # DBCs don't need an old state
            elif "file" in d.keys():
                assert "curve" not in d.keys() and "val" not in d.keys() and "expression" not in d.keys() and "fileseries" not in d.keys()
                fle = d["file"]  # a single file
                ftype = d.get("ftype", "id_val")
                # to ramp the file by a time curve
                ramp_curve = d.get("ramp_curve", None)
                if ramp_curve is not None:
                    func_ramp, func_file = (
                        fem.Function(V_dbc),
                        fem.Function(V_dbc),
                    )
                    # first read file into function
                    self.pb.io.readfunction(func_file, fle)
                    # now store ramp curve into function
                    load_ = expression.template_vector(dim=self.dim)
                    load_.val_x, load_.val_y, load_.val_z = (
                        self.pb.ti.timecurves(d["ramp_curve"])(self.pb.ti.t_init),
                        self.pb.ti.timecurves(d["ramp_curve"])(self.pb.ti.t_init),
                        self.pb.ti.timecurves(d["ramp_curve"])(self.pb.ti.t_init),
                    )
                    func_ramp.interpolate(load_.evaluate)
                    func_ramp.x.petsc_vec.ghostUpdate(
                        addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD,
                    )
                    self.pb.ti.funcs_to_update_vec.append(
                        {
                            func_ramp: [
                                self.pb.ti.timecurves(d["ramp_curve"]),
                                self.pb.ti.timecurves(d["ramp_curve"]),
                                self.pb.ti.timecurves(d["ramp_curve"]),
                            ],
                            "funcs_mult": [func_file, func],
                        }
                    )
                    self.pb.ti.funcs_to_update_vec_old.append({None: -1})  # DBCs don't need an old state
                    # now multiply
                    func.x.petsc_vec.pointwiseMult(func_ramp.x.petsc_vec, func_file.x.petsc_vec)
                    func.x.petsc_vec.ghostUpdate(
                        addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD,
                    )
                else:
                    # read file into function
                    self.pb.io.readfunction(func, fle)
            elif "fileseries" in d.keys():  # file series, where we'd have one file per time step
                assert "curve" not in d.keys() and "val" not in d.keys() and "expression" not in d.keys() and "file" not in d.keys()
                scale = d.get("scale", 1.0)
                self.pb.ti.funcs_data.append({func: d["fileseries"], "scale": scale})
                self.have_dirichlet_fileseries = True
            else:
                raise RuntimeError("Need to have 'curve', 'val', 'expression', 'file', or 'fileseries' specified!")

            if d["dir"] == "all":
                if "id" in d.keys():
                    if all(isinstance(x, int) for x in d["id"]):
                        nodes_bc = fem.locate_dofs_topological(V_dbc, codim, mdata.indices[np.isin(mdata.values, d["id"])])
                    else:
                        nodes_bc_ = []
                        for lc in d["id"]:
                            nodes_bc_.append(fem.locate_dofs_geometrical(V_dbc, lc.evaluate))
                        nodes_bc = np.concatenate(nodes_bc_).ravel()
                else:
                    cells = mesh.locate_entities(self.pb.mesh, codim, self.all)
                    nodes_bc = fem.locate_dofs_topological(V_dbc, codim, cells)
                dbcs.append(fem.dirichletbc(func, nodes_bc))

            elif d["dir"] == "x":
                if "id" in d.keys():
                    if all(isinstance(x, int) for x in d["id"]):
                        nodes_bc_x = fem.locate_dofs_topological(V_dbc.sub(0),codim, mdata.indices[np.isin(mdata.values, d["id"])])
                    else:
                        nodes_bc_x_ = []
                        for lc in d["id"]:
                            nodes_bc_x_.append(fem.locate_dofs_geometrical((V_dbc.sub(0), V_dbc.sub(0).collapse()[0]), lc.evaluate)[0])
                        nodes_bc_x = np.concatenate(nodes_bc_x_).ravel()
                else:
                    cells = mesh.locate_entities(self.pb.mesh, codim, self.all)
                    nodes_bc_x = fem.locate_dofs_topological(V_dbc.sub(0), codim, cells)
                dbcs.append(fem.dirichletbc(func.sub(0), nodes_bc_x))

            elif d["dir"] == "y":
                if "id" in d.keys():
                    if all(isinstance(x, int) for x in d["id"]):
                        nodes_bc_y = fem.locate_dofs_topological(V_dbc.sub(1),codim, mdata.indices[np.isin(mdata.values, d["id"])])
                    else:
                        nodes_bc_y_ = []
                        for lc in d["id"]:
                            nodes_bc_y_.append(fem.locate_dofs_geometrical((V_dbc.sub(1), V_dbc.sub(1).collapse()[0]), lc.evaluate)[0])
                        nodes_bc_y = np.concatenate(nodes_bc_y_).ravel()
                else:
                    cells = mesh.locate_entities(self.pb.mesh, codim, self.all)
                    nodes_bc_y = fem.locate_dofs_topological(V_dbc.sub(1), codim, cells)
                dbcs.append(fem.dirichletbc(func.sub(1), nodes_bc_y))

            elif d["dir"] == "z":
                if "id" in d.keys():
                    if all(isinstance(x, int) for x in d["id"]):
                        nodes_bc_z = fem.locate_dofs_topological(V_dbc.sub(2),codim, mdata.indices[np.isin(mdata.values, d["id"])])
                    else:
                        nodes_bc_z_ = []
                        for lc in d["id"]:
                            nodes_bc_z_.append(fem.locate_dofs_geometrical((V_dbc.sub(2), V_dbc.sub(2).collapse()[0]), lc.evaluate)[0])
                        nodes_bc_z = np.concatenate(nodes_bc_z_).ravel()
                else:
                    cells = mesh.locate_entities(self.pb.mesh, codim, self.all)
                    nodes_bc_z = fem.locate_dofs_topological(V_dbc.sub(2), codim, cells)
                dbcs.append(fem.dirichletbc(func.sub(2), nodes_bc_z))

            elif d["dir"] == "all_by_dofs":
                dbcs.append(fem.dirichletbc(func, d["dofs"]))

            elif d["dir"] == "x_by_dofs":
                dbcs.append(fem.dirichletbc(func.sub(0), d["dofs"]))

            elif d["dir"] == "y_by_dofs":
                dbcs.append(fem.dirichletbc(func.sub(1), d["dofs"]))

            elif d["dir"] == "z_by_dofs":
                dbcs.append(fem.dirichletbc(func.sub(2), d["dofs"]))

            else:
                raise NameError("Unknown dir option for Dirichlet BC!")

    # function that marks all dofs
    def all(self, x):
        return np.full(x.shape[1], True, dtype=bool)

    # set Neumann BCs
    def neumann_bcs(
        self,
        bcdict,
        ds_,
        F=None,
        funcs_to_update=None,
        funcs_to_update_vec=None,
        funcsexpr_to_update=None,
        funcsexpr_to_update_vec=None,
    ):
        w = ufl.as_ufl(0)

        for n in bcdict:
            codim = n.get("codimension", self.dim - 1)
            assert(codim==self.dim - 1) # currently, only integration on codimension dim-1 supported (in a straightforward way...)
            ID, dind = "id", 0
            if "is_locator" in n.keys(): dind=2
            if "id_loc" in n.keys(): ID="id_loc"

            if n["dir"] == "xyz_ref":  # reference xyz
                func = fem.Function(self.V_field)

                if "curve" in n.keys():
                    assert "val" not in n.keys() and "expression" not in n.keys()
                    load = expression.template_vector(dim=self.dim)
                    load.val_x, load.val_y, load.val_z = (
                        self.pb.ti.timecurves(n["curve"][0])(self.pb.ti.t_init),
                        self.pb.ti.timecurves(n["curve"][1])(self.pb.ti.t_init),
                        self.pb.ti.timecurves(n["curve"][2])(self.pb.ti.t_init),
                    )
                    func.interpolate(load.evaluate)
                    func.x.petsc_vec.ghostUpdate(
                        addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD,
                    )
                    funcs_to_update_vec.append(
                        {
                            func: [
                                self.pb.ti.timecurves(n["curve"][0]),
                                self.pb.ti.timecurves(n["curve"][1]),
                                self.pb.ti.timecurves(n["curve"][2]),
                            ]
                        }
                    )
                elif "val" in n.keys():
                    assert "curve" not in n.keys() and "expression" not in n.keys()
                    func.x.petsc_vec.set(
                        n["val"]
                    )  # currently only one value for all directions - use constant load function otherwise!
                elif "expression" in n.keys():
                    assert "curve" not in n.keys() and "val" not in n.keys()
                    expr = n["expression"]()
                    expr.t = self.pb.ti.t_init
                    func.interpolate(expr.evaluate)
                    func.x.petsc_vec.ghostUpdate(
                        addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD,
                    )
                    funcsexpr_to_update_vec[func] = expr
                else:
                    raise RuntimeError("Need to have 'curve', 'val', or 'expression' specified!")

                for i in range(len(n[ID])):
                    w += self.pb.vf.deltaW_ext_neumann_ref(func, ds_[dind](n[ID][i]))

            elif n["dir"] == "normal_ref":  # reference normal
                func = fem.Function(self.Vdisc_scalar)

                if "curve" in n.keys():
                    assert "val" not in n.keys() and "expression" not in n.keys()
                    load = expression.template()
                    load.val = self.pb.ti.timecurves(n["curve"])(self.pb.ti.t_init)
                    func.interpolate(load.evaluate)
                    func.x.petsc_vec.ghostUpdate(
                        addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD,
                    )
                    funcs_to_update.append({func: self.pb.ti.timecurves(n["curve"])})
                elif "val" in n.keys():
                    assert "curve" not in n.keys() and "expression" not in n.keys()
                    func.x.petsc_vec.set(n["val"])
                elif "expression" in n.keys():
                    assert "curve" not in n.keys() and "val" not in n.keys()
                    expr = n["expression"]()
                    expr.t = self.pb.ti.t_init
                    func.interpolate(expr.evaluate)
                    func.x.petsc_vec.ghostUpdate(
                        addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD,
                    )
                    funcsexpr_to_update[func] = expr
                else:
                    raise RuntimeError("Need to have 'curve', 'val', or 'expression' specified!")

                for i in range(len(n[ID])):
                    w += self.pb.vf.deltaW_ext_neumann_normal_ref(func, ds_[dind](n[ID][i]))

            elif n["dir"] == "xyz_cur":  # current xyz
                func = fem.Function(self.V_field)

                if "curve" in n.keys():
                    assert "val" not in n.keys() and "expression" not in n.keys()
                    load = expression.template_vector(dim=self.dim)
                    load.val_x, load.val_y, load.val_z = (
                        self.pb.ti.timecurves(n["curve"][0])(self.pb.ti.t_init),
                        self.pb.ti.timecurves(n["curve"][1])(self.pb.ti.t_init),
                        self.pb.ti.timecurves(n["curve"][2])(self.pb.ti.t_init),
                    )
                    func.interpolate(load.evaluate)
                    func.x.petsc_vec.ghostUpdate(
                        addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD,
                    )
                    funcs_to_update_vec.append(
                        {
                            func: [
                                self.pb.ti.timecurves(n["curve"][0]),
                                self.pb.ti.timecurves(n["curve"][1]),
                                self.pb.ti.timecurves(n["curve"][2]),
                            ]
                        }
                    )
                elif "val" in n.keys():
                    assert "curve" not in n.keys() and "expression" not in n.keys()
                    func.x.petsc_vec.set(
                        n["val"]
                    )  # currently only one value for all directions - use constant load function otherwise!
                elif "expression" in n.keys():
                    assert "curve" not in n.keys() and "val" not in n.keys()
                    expr = n["expression"]()
                    expr.t = self.pb.ti.t_init
                    func.interpolate(expr.evaluate)
                    func.x.petsc_vec.ghostUpdate(
                        addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD,
                    )
                    funcsexpr_to_update_vec[func] = expr
                else:
                    raise RuntimeError("Need to have 'curve', 'val', or 'expression' specified!")

                for i in range(len(n[ID])):
                    w += self.pb.vf.deltaW_ext_neumann_cur(func, ds_[dind](n[ID][i]), F=F)

            elif n["dir"] == "normal_cur":  # current normal
                func = fem.Function(self.Vdisc_scalar)

                if "curve" in n.keys():
                    assert "val" not in n.keys() and "expression" not in n.keys()
                    load = expression.template()
                    load.val = self.pb.ti.timecurves(n["curve"])(self.pb.ti.t_init)
                    func.interpolate(load.evaluate)
                    func.x.petsc_vec.ghostUpdate(
                        addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD,
                    )
                    funcs_to_update.append({func: self.pb.ti.timecurves(n["curve"])})
                elif "val" in n.keys():
                    assert "curve" not in n.keys() and "expression" not in n.keys()
                    func.x.petsc_vec.set(n["val"])
                elif "expression" in n.keys():
                    assert "curve" not in n.keys() and "val" not in n.keys()
                    expr = n["expression"]()
                    expr.t = self.pb.ti.t_init
                    func.interpolate(expr.evaluate)
                    func.x.petsc_vec.ghostUpdate(
                        addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD,
                    )
                    funcsexpr_to_update[func] = expr
                else:
                    raise RuntimeError("Need to have 'curve', 'val', or 'expression' specified!")

                for i in range(len(n[ID])):
                    w += self.pb.vf.deltaW_ext_neumann_normal_cur(func, ds_[dind](n[ID][i]), F=F)

            else:
                raise NameError("Unknown dir option for Neumann BC!")

        return w

    # set Neumann BCs for prestress
    def neumann_prestress_bcs(
        self,
        bcdict,
        ds_,
        funcs_to_update=None,
        funcs_to_update_vec=None,
        funcsexpr_to_update=None,
        funcsexpr_to_update_vec=None,
    ):
        w = ufl.as_ufl(0)

        for n in bcdict:
            codim = n.get("codimension", self.dim - 1)
            assert(codim==self.dim - 1) # currently, only integration on codimension dim-1 supported (in a straightforward way...)
            ID, dind = "id", 0
            if "is_locator" in n.keys(): dind=2
            if "id_loc" in n.keys(): ID="id_loc"

            if n["dir"] == "xyz_ref":  # reference xyz
                func = fem.Function(self.V_field)

                if "curve" in n.keys():
                    assert "val" not in n.keys() and "expression" not in n.keys()
                    load = expression.template_vector(dim=self.dim)
                    load.val_x, load.val_y, load.val_z = (
                        self.pb.ti.timecurves(n["curve"][0])(self.pb.ti.t_init),
                        self.pb.ti.timecurves(n["curve"][1])(self.pb.ti.t_init),
                        self.pb.ti.timecurves(n["curve"][2])(self.pb.ti.t_init),
                    )
                    func.interpolate(load.evaluate)
                    func.x.petsc_vec.ghostUpdate(
                        addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD,
                    )
                    funcs_to_update_vec.append(
                        {
                            func: [
                                self.pb.ti.timecurves(n["curve"][0]),
                                self.pb.ti.timecurves(n["curve"][1]),
                                self.pb.ti.timecurves(n["curve"][2]),
                            ]
                        }
                    )
                elif "val" in n.keys():
                    assert "curve" not in n.keys() and "expression" not in n.keys()
                    func.x.petsc_vec.set(
                        n["val"]
                    )  # currently only one value for all directions - use constant load function otherwise!
                elif "expression" in n.keys():
                    assert "curve" not in n.keys() and "val" not in n.keys()
                    expr = n["expression"]()
                    expr.t = self.pb.ti.t_init
                    func.interpolate(expr.evaluate)
                    func.x.petsc_vec.ghostUpdate(
                        addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD,
                    )
                    funcsexpr_to_update_vec[func] = expr
                else:
                    raise RuntimeError("Need to have 'curve', 'val', or 'expression' specified!")

                for i in range(len(n[ID])):
                    w += self.pb.vf.deltaW_ext_neumann_ref(func, ds_[dind](n[ID][i]))

            elif n["dir"] == "normal_ref":  # reference normal
                func = fem.Function(self.Vdisc_scalar)

                if "curve" in n.keys():
                    assert "val" not in n.keys() and "expression" not in n.keys()
                    load = expression.template()
                    load.val = self.pb.ti.timecurves(n["curve"])(self.pb.ti.t_init)
                    func.interpolate(load.evaluate)
                    func.x.petsc_vec.ghostUpdate(
                        addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD,
                    )
                    funcs_to_update.append({func: self.pb.ti.timecurves(n["curve"])})
                elif "val" in n.keys():
                    assert "curve" not in n.keys() and "expression" not in n.keys()
                    func.x.petsc_vec.set(
                        n["val"]
                    )  # currently only one value for all directions - use constant load function otherwise!
                elif "expression" in n.keys():
                    assert "curve" not in n.keys() and "val" not in n.keys()
                    expr = n["expression"]()
                    expr.t = self.pb.ti.t_init
                    func.interpolate(expr.evaluate)
                    func.x.petsc_vec.ghostUpdate(
                        addv=PETSc.InsertMode.INSERT,
                        mode=PETSc.ScatterMode.FORWARD,
                    )
                    funcsexpr_to_update[func] = expr
                else:
                    raise RuntimeError("Need to have 'curve', 'val', or 'expression' specified!")

                for i in range(len(n[ID])):
                    w += self.pb.vf.deltaW_ext_neumann_normal_ref(func, ds_[dind](n[ID][i]))

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
            codim = r.get("codimension", self.dim - 1)
            assert(codim==self.dim - 1) # currently, only integration on codimension dim-1 supported (in a straightforward way...)
            ID, dind = "id", 0
            if "is_locator" in r.keys(): dind=2
            if "id_loc" in r.keys(): ID="id_loc"

            direction = r.get("dir", "xyz_ref")

            if r["type"] == "spring":
                # may be an expression
                stiff = r["stiff"]
                if inspect.isclass(stiff):
                    stiff_expr = stiff()
                    stiff_ = fem.Function(self.Vdisc_scalar)
                    stiff_.interpolate(stiff_expr.evaluate)
                else:
                    stiff_ = stiff

                if direction == "xyz_ref":  # reference xyz
                    for i in range(len(r[ID])):
                        w += self.pb.vf.deltaW_ext_robin_spring(u, stiff_, ds_[dind](r[ID][i]), u_pre)

                elif direction == "normal_ref":  # reference normal
                    for i in range(len(r[ID])):
                        w += self.pb.vf.deltaW_ext_robin_spring_normal_ref(u, stiff_, ds_[dind](r[ID][i]), u_pre)

                elif direction == "normal_cross":  # cross normal
                    for i in range(len(r[ID])):
                        w += self.pb.vf.deltaW_ext_robin_spring_normal_cross(u, stiff_, ds_[dind](r[ID][i]), u_pre)

                else:
                    raise NameError("Unknown dir option for Robin BC!")

            elif r["type"] == "dashpot":
                # may be an expression
                visc = r["visc"]
                if inspect.isclass(visc):
                    visc_expr = visc()
                    visc_ = fem.Function(self.Vdisc_scalar)
                    visc_.interpolate(visc_expr.evaluate)
                else:
                    visc_ = visc

                if direction == "xyz_ref":  # reference xyz
                    for i in range(len(r[ID])):
                        w += self.pb.vf.deltaW_ext_robin_dashpot(v, visc_, ds_[dind](r[ID][i]))

                elif direction == "normal_ref":  # reference normal
                    for i in range(len(r[ID])):
                        w += self.pb.vf.deltaW_ext_robin_dashpot_normal_ref(v, visc_, ds_[dind](r[ID][i]))

                elif direction == "normal_cross":  # cross normal
                    for i in range(len(r[ID])):
                        w += self.pb.vf.deltaW_ext_robin_dashpot_normal_cross(v, visc_, ds_[dind](r[ID][i]))

                else:
                    raise NameError("Unknown dir option for Robin BC!")

            else:
                raise NameError("Unknown type option for Robin BC!")

        return w

    # set membrane surface BCs
    def membranesurf_bcs(self, bcdict, u, v, a, ds_, ivar=None, wallfields=[], actweights=[]):
        w, idmem, bstress, bstrainenergy, bintpower = (
            ufl.as_ufl(0),
            [],
            [],
            [],
            [],
        )

        mi = 0
        for m in bcdict:
            codim = m.get("codimension", self.dim - 1)
            assert(codim==self.dim - 1) # currently, only integration on codimension dim-1 supported (in a straightforward way...)
            ID, dind = "id", 0
            if "is_locator" in m.keys(): dind=2
            if "id_loc" in m.keys(): ID="id_loc"

            internal = m.get("internal", False)

            if internal:
                dind = 1
                fcts = m.get("facet_side", "+")
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

            for i in range(len(m[ID])):
                idmem.append(m[ID][i])

                w += self.pb.vf.deltaW_ext_membrane(
                    self.pb.ki.F(u),
                    self.pb.ki.Fdot(v),
                    a,
                    m["params"],
                    ds_[dind](m[ID][i]),
                    ivar=ivar,
                    fibfnc=self.pb.fib_func,
                    wallfield=wallfield,
                    actweight=actweight,
                    fcts=fcts,
                )
                bstr, bse, bip = self.pb.vf.deltaW_ext_membrane(
                    self.pb.ki.F(u),
                    self.pb.ki.Fdot(v),
                    a,
                    m["params"],
                    ds_[dind](m[ID][i]),
                    ivar=ivar,
                    fibfnc=self.pb.fib_func,
                    wallfield=wallfield,
                    actweight=actweight,
                    fcts=fcts,
                    returnquantity="stress_energy_power",
                )
                bstress.append(bstr)
                bstrainenergy.append(bse)
                bintpower.append(bip)

            mi += 1

        return w, idmem, bstress, bstrainenergy, bintpower

    # set body forces (technically, no "boundary" conditions, since acting on a volume element... but implemented here for convenience)
    def bodyforce(self, mdict, dx_, rho, F=None, chi=None, funcs_to_update=None, funcsexpr_to_update=None):
        func, func_dir = (
            fem.Function(self.Vdisc_scalar),
            fem.Function(self.V_field),
        )

        # direction needs to be set
        driection = expression.template_vector(dim=self.dim)
        dir_x, dir_y, dir_z = mdict["dir"][0], mdict["dir"][1], mdict["dir"][2]
        dir_norm = np.sqrt(dir_x**2.0 + dir_y**2.0 + dir_z**2.0)
        driection.val_x, driection.val_y, driection.val_z = (
            dir_x / dir_norm,
            dir_y / dir_norm,
            dir_z / dir_norm,
        )
        func_dir.interpolate(driection.evaluate)

        if "curve" in mdict.keys():
            assert "val" not in mdict.keys() and "expression" not in mdict.keys()
            load = expression.template()
            load.val = self.pb.ti.timecurves(mdict["curve"])(self.pb.ti.t_init)
            func.interpolate(load.evaluate)
            func.x.petsc_vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT,
                mode=PETSc.ScatterMode.FORWARD,
            )
            funcs_to_update.append({func: self.pb.ti.timecurves(mdict["curve"])})
        elif "val" in mdict.keys():
            assert "curve" not in mdict.keys() and "expression" not in mdict.keys()
            func.x.petsc_vec.set(mdict["val"])
        elif "expression" in mdict.keys():
            assert "curve" not in mdict.keys() and "val" not in mdict.keys()
            expr = mdict["expression"]()
            expr.t = self.pb.ti.t_init
            func.interpolate(expr.evaluate)
            func.x.petsc_vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT,
                mode=PETSc.ScatterMode.FORWARD,
            )
            funcsexpr_to_update[func] = expr
        else:
            raise RuntimeError("Need to have 'curve', 'val', or 'expression' specified!")

        # scale by density
        scale_dens = mdict.get("scale_density", False)

        return self.pb.vf.deltaW_ext_bodyforce(func, func_dir, rho, dx_, F=F, chi=chi, scale_dens=scale_dens)



class boundary_cond_fluid(boundary_cond):
    # set stabilized Neumann BCs
    def stabilized_neumann_bcs(self, bcdict, v, ds_, wel=None, F=None):
        w = ufl.as_ufl(0)

        for sn in bcdict:
            codim = sn.get("codimension", self.dim - 1)
            assert(codim==self.dim - 1) # currently, only integration on codimension dim-1 supported (in a straightforward way...)
            ID, dind = "id", 0
            if "is_locator" in sn.keys(): dind=2
            if "id_loc" in sn.keys(): ID="id_loc"

            for i in range(len(sn[ID])):
                beta = sn["beta"]

                w += self.pb.vf.deltaW_ext_stabilized_neumann(v, beta, ds_[dind](sn[ID][i]), w=wel, F=F)

        return w

    # set mod. stabilized Neumann BCs
    def stabilized_neumann_mod_bcs(self, bcdict, v, ds_, wel=None, F=None):
        w = ufl.as_ufl(0)

        for sn in bcdict:
            codim = sn.get("codimension", self.dim - 1)
            assert(codim==self.dim - 1) # currently, only integration on codimension dim-1 supported (in a straightforward way...)
            ID, dind = "id", 0
            if "is_locator" in sn.keys(): dind=2
            if "id_loc" in sn.keys(): ID="id_loc"

            for i in range(len(sn[ID])):
                beta = sn["beta"]
                gamma = sn["gamma"]

                w += self.pb.vf.deltaW_ext_stabilized_neumann_mod(v, beta, gamma, ds_[dind](sn[ID][i]), w=wel, F=F)

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

            codim = r.get("codimension", self.dim - 1)
            assert(codim==self.dim - 1) # currently, only integration on codimension dim-1 supported (in a straightforward way...)
            ID, dind = "id", 0
            if "is_locator" in r.keys(): dind=2
            if "id_loc" in r.keys(): ID="id_loc"

            direction = r.get("dir", "xyz_ref")

            beta_.append(fem.Function(self.Vdisc_scalar))

            if direction == "xyz_ref":  # reference xyz
                for i in range(len(r[ID])):
                    if dw is None:
                        w += self.pb.vf.deltaW_ext_robin_valve(
                            v,
                            beta_[-1],
                            dS_[0](r[ID][i]),
                            fcts="+",
                            w=wel_,
                            F=F,
                        )
                    else:
                        # derivative (for implicit valve law)
                        dwddp += self.pb.vf.deltaW_ext_robin_valve(
                            v,
                            ufl.as_ufl(1.0),
                            dS_[0](r[ID][i]),
                            fcts="+",
                            w=wel_,
                            F=F,
                        )

            elif direction == "normal_ref":  # reference normal
                for i in range(len(r[ID])):
                    if dw is None:
                        w += self.pb.vf.deltaW_ext_robin_valve_normal_ref(
                            v,
                            beta_[-1],
                            dS_[0](r[ID][i]),
                            fcts="+",
                            w=wel_,
                            F=F,
                        )
                    else:
                        # derivative (for implicit valve law)
                        dwddp += self.pb.vf.deltaW_ext_robin_valve_normal_ref(
                            v,
                            ufl.as_ufl(1.0),
                            dS_[0](r[ID][i]),
                            fcts="+",
                            w=wel_,
                            F=F,
                        )

            else:
                raise NameError("Unknown dir option for Robin valve BC!")

            # one dwddp term per valve
            if dw is not None:
                dw.append(dwddp)

        return w

    # set flux monitor conditions
    def flux_monitor_bcs(self, bcdict, v, qdict_, wel=None, F=None):
        if wel is None:
            wel_ = ufl.constantvalue.zero(self.dim)
        else:
            wel_ = wel

        for r in bcdict:
            codim = r.get("codimension", self.dim - 1)
            assert(codim==self.dim - 1) # currently, only integration on codimension dim-1 supported (in a straightforward way...)
            ID, dind = "id", 0
            if "is_locator" in r.keys(): dind=2
            if "id_loc" in r.keys(): ID="id_loc"

            q = ufl.as_ufl(0)

            internal = r.get("internal", False)
            on_subdomain = r.get("on_subdomain", False)

            if internal:
                assert not on_subdomain
                fcts = r.get("facet_side", "+")
            else:
                fcts = None

            if on_subdomain:
                dom_u = r["domain"]

            for i in range(len(r[ID])):
                if not internal:
                    if not on_subdomain:
                        db_ = ufl.ds(
                            domain=self.pb.io.mesh_master,
                            subdomain_data=self.pb.io.mt_b_master,
                            subdomain_id=r[ID][i],
                            metadata={"quadrature_degree": self.pb.quad_degree},
                        )
                    else:
                        db_ = ufl.ds(
                            domain=self.pb.io.submshes_emap[dom_u][0],
                            subdomain_data=self.pb.io.sub_mt_b[dom_u],
                            subdomain_id=r[ID][i],
                            metadata={"quadrature_degree": self.pb.quad_degree},
                        )
                else:
                    db_ = ufl.dS(
                        domain=self.pb.io.mesh_master,
                        subdomain_data=self.pb.io.mt_b_master,
                        subdomain_id=r[ID][i],
                        metadata={"quadrature_degree": self.pb.quad_degree},
                    )

                q += self.pb.vf.flux(v, db_, w=wel_, F=F, fcts=fcts)

            if on_subdomain:
                # entity map child to parent
                em_u = [self.pb.io.submshes_emap[dom_u][1]]
                qdict_.append(fem.form(q, entity_maps=em_u))
            else:
                qdict_.append(fem.form(q))

    # set dp monitor conditions
    def dp_monitor_bcs(self, bcdict, a_u_, a_d_, pint_u_, pint_d_, pdict, F=None):
        for r in bcdict:
            codim = r.get("codimension", self.dim - 1)
            assert(codim==self.dim - 1) # currently, only integration on codimension dim-1 supported (in a straightforward way...)
            ID, dind = "id", 0
            if "is_locator" in r.keys(): dind=2
            if "id_loc" in r.keys(): ID="id_loc"

            spatial = r.get("spatial", False)

            # area map for integration
            if spatial and F is not None:
                J = ufl.det(F)
                ja = J * ufl.sqrt(ufl.dot(self.pb.vf.n0, (ufl.inv(F) * ufl.inv(F).T) * self.pb.vf.n0))
            else:
                ja = 1.0

            dom_u, dom_d = r["upstream_domain"], r["downstream_domain"]

            a_u, a_d, pint_u, pint_d = (
                ufl.as_ufl(0),
                ufl.as_ufl(0),
                ufl.as_ufl(0),
                ufl.as_ufl(0),
            )

            for i in range(len(r[ID])):
                db_u_ = ufl.ds(
                    domain=self.pb.io.submshes_emap[dom_u][0],
                    subdomain_data=self.pb.io.sub_mt_b[dom_u],
                    subdomain_id=r[ID][i],
                    metadata={"quadrature_degree": self.pb.quad_degree},
                )
                db_d_ = ufl.ds(
                    domain=self.pb.io.submshes_emap[dom_d][0],
                    subdomain_data=self.pb.io.sub_mt_b[dom_d],
                    subdomain_id=r[ID][i],
                    metadata={"quadrature_degree": self.pb.quad_degree},
                )

                # area forms
                a_u += ja * db_u_
                a_d += ja * db_d_

                # pressure forms
                pint_u += pdict[dom_u] * ja * db_u_
                pint_d += pdict[dom_d] * ja * db_d_

            # entity maps child to parent
            em_u = [self.pb.io.submshes_emap[dom_u][1]]
            em_d = [self.pb.io.submshes_emap[dom_d][1]]

            a_u_.append(fem.form(a_u, entity_maps=em_u))
            a_d_.append(fem.form(a_d, entity_maps=em_d))

            pint_u_.append(fem.form(pint_u, entity_maps=em_u))
            pint_d_.append(fem.form(pint_d, entity_maps=em_d))
