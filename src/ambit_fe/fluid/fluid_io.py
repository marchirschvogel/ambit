#!/usr/bin/env python3

# Copyright (c) 2019-2026, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

from dolfinx import fem, io
from petsc4py import PETSc
import ufl
from ..solver.projection import project
from ..ioroutines import IO_field


class IO_fluid(IO_field):
    def __init__(self, pb):
        self.pb = pb
        self.results_pre = []

    def write_output(self, writemesh=False, N=1, t=0):
        if self.pb.io.indicate_results_by == "time":
            indicator = t
        elif self.pb.io.indicate_results_by == "step":
            indicator = N
        elif self.pb.io.indicate_results_by == "step0":
            if self.pb.io.write_results_every > 0:
                indicator = int(N / self.pb.io.write_results_every) - 1
            else:
                indicator = 0
        else:
            raise ValueError("Unknown indicate_results_by optin. Choose 'time' or 'step'.")

        if writemesh:
            if self.pb.io.write_results_every > 0:
                for res in self.pb.results_to_write:
                    if res not in self.results_pre:
                        if res == "pressure" and bool(self.pb.io.duplicate_mesh_domains):
                            for m, mp in enumerate(self.pb.io.duplicate_mesh_domains):
                                outfile = io.XDMFFile(
                                    self.pb.comm,
                                    self.pb.io.output_path
                                    + "/results_"
                                    + self.pb.pbase.simname
                                    + "_"
                                    + self.pb.problem_physics
                                    + "_"
                                    + res
                                    + str(m + 1)
                                    + ".xdmf",
                                    "w",
                                )
                                outfile.write_mesh(self.pb.io.submshes_emap[m + 1][0])
                                self.pb.resultsfiles[res + str(m + 1)] = outfile
                        else:
                            outfile = io.XDMFFile(
                                self.pb.comm,
                                self.pb.io.output_path
                                + "/results_"
                                + self.pb.pbase.simname
                                + "_"
                                + self.pb.problem_physics
                                + "_"
                                + res
                                + ".xdmf",
                                "w",
                            )
                            outfile.write_mesh(self.pb.mesh)
                            self.pb.resultsfiles[res] = outfile

            return

        else:
            # write results every write_results_every steps
            if self.pb.io.write_results_every > 0 and N % self.pb.io.write_results_every == 0:
                # save solution to XDMF format
                for res in self.pb.results_to_write:
                    if res == "velocity":
                        if self.pb.io.output_midpoint:
                            v_proj = project(
                                self.pb.vel_mid,
                                self.pb.V_v,
                                self.pb.dx,
                                domids=self.pb.domain_ids,
                                nm="Velocity",
                                comm=self.pb.comm,
                                entity_maps=self.pb.io.entity_maps,
                            )
                            v_out = fem.Function(self.pb.V_out_vector, name=v_proj.name)
                            v_out.interpolate(v_proj)
                        else:
                            v_out = fem.Function(self.pb.V_out_vector, name=self.pb.v.name)
                            v_out.interpolate(self.pb.v)
                        self.pb.resultsfiles[res].write_function(v_out, indicator)
                    elif res == "acceleration":  # passed in a is not a function but form, so we have to project
                        if self.pb.io.output_midpoint:
                            acc = self.pb.acc_mid
                        else:
                            acc = self.pb.acc
                        a_proj = project(
                            acc,
                            self.pb.V_v,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="Acceleration",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        a_out = fem.Function(self.pb.V_out_vector, name=a_proj.name)
                        a_out.interpolate(a_proj)
                        self.pb.resultsfiles[res].write_function(a_out, indicator)
                    elif res == "pressure":
                        if bool(self.pb.io.duplicate_mesh_domains):
                            for m, mp in enumerate(self.pb.io.duplicate_mesh_domains):
                                V_out_scalar_sub = fem.functionspace(
                                    self.pb.io.submshes_emap[m + 1][0],
                                    ("Lagrange", self.pb.mesh_degree),
                                )
                                if self.pb.io.output_midpoint:
                                    p_proj = project(
                                        self.pb.pf_mid_[m],
                                        self.pb.V_p_[m],
                                        self.pb.dx_p[m],
                                        domids=[self.pb.domain_ids[m]],
                                        nm=self.pb.p_[m].name,
                                        comm=self.pb.comm,
                                        entity_maps=[self.pb.io.submshes_emap[m + 1][1]],
                                    )
                                    p_out = fem.Function(V_out_scalar_sub, name=p_proj.name)
                                    p_out.interpolate(p_proj)
                                else:
                                    p_out = fem.Function(V_out_scalar_sub, name=self.pb.p_[m].name)
                                    p_out.interpolate(self.pb.p_[m])
                                self.pb.resultsfiles[res + str(m + 1)].write_function(p_out, indicator)
                        else:
                            if self.pb.io.output_midpoint:
                                p_proj = project(
                                    self.pb.pf_mid_[0],
                                    self.pb.V_p_[0],
                                    self.pb.dx,
                                    domids=self.pb.domain_ids,
                                    nm=self.pb.p_[0].name,
                                    comm=self.pb.comm,
                                    entity_maps=self.pb.io.entity_maps,
                                )
                                p_out = fem.Function(self.pb.V_out_scalar, name=p_proj.name)
                                p_out.interpolate(p_proj)
                            else:
                                p_out = fem.Function(self.pb.V_out_scalar, name=self.pb.p_[0].name)
                                p_out.interpolate(self.pb.p_[0])
                            self.pb.resultsfiles[res].write_function(p_out, indicator)
                    elif res == "cauchystress":
                        stressfuncs = []
                        for n in range(self.pb.num_domains):
                            if self.pb.io.output_midpoint:
                                v, p, F, chi = self.pb.vel_mid, self.pb.pf_mid_[n], self.pb.alevar["Fale_mid"], self.pb.phasevar["chi_mid"]
                            else:
                                v, p, F, chi = self.pb.v, self.pb.p_[n], self.pb.alevar["Fale"], self.pb.phasevar["chi"]
                            stressfuncs.append(self.pb.ma[n].sigma(v, p, F=F, chi=chi))
                        cauchystress = project(
                            stressfuncs,
                            self.pb.Vd_tensor,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="CauchyStress",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        cauchystress_out = fem.Function(self.pb.V_out_tensor, name=cauchystress.name)
                        cauchystress_out.interpolate(cauchystress)
                        self.pb.resultsfiles[res].write_function(cauchystress_out, indicator)
                    elif res == "fluiddisplacement":  # passed in uf is not a function but form, so we have to project
                        if self.pb.io.output_midpoint:
                            uf = self.pb.ufluid_mid
                        else:
                            uf = self.pb.ufluid
                        uf_proj = project(
                            uf,
                            self.pb.V_v,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="FluidDisplacement",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        uf_out = fem.Function(self.pb.V_out_vector, name=uf_proj.name)
                        uf_out.interpolate(uf_proj)
                        self.pb.resultsfiles[res].write_function(uf_out, indicator)
                    elif res == "fibers":
                        # written only once at the beginning, not after each time step (since constant in time)
                        pass
                    elif res == "cauchystress_membrane":
                        stressfuncs = []
                        for n in range(len(self.pb.bstress)):
                            stressfuncs.append(self.pb.bstress[n])
                        cauchystress_membrane = project(
                            stressfuncs,
                            self.pb.Vd_tensor,
                            self.pb.bmeasures[0],
                            domids=self.pb.idmem,
                            nm="CauchyStress_membrane",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        cauchystress_membrane_out = fem.Function(self.pb.V_out_tensor, name=cauchystress_membrane.name)
                        cauchystress_membrane_out.interpolate(cauchystress_membrane)
                        self.pb.resultsfiles[res].write_function(cauchystress_membrane_out, indicator)
                    elif res == "strainenergy_membrane":
                        sefuncs = []
                        for n in range(len(self.pb.bstrainenergy)):
                            sefuncs.append(self.pb.bstrainenergy[n])
                        strainenergy_membrane = project(
                            sefuncs,
                            self.pb.Vd_scalar,
                            self.pb.bmeasures[0],
                            domids=self.pb.idmem,
                            nm="StrainEnergy_membrane",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        strainenergy_membrane_out = fem.Function(self.pb.V_out_scalar, name=strainenergy_membrane.name)
                        strainenergy_membrane_out.interpolate(strainenergy_membrane)
                        self.pb.resultsfiles[res].write_function(strainenergy_membrane_out, indicator)
                    elif res == "density":
                        if self.pb.io.output_midpoint:
                            chi = self.pb.phasevar["chi_mid"]
                        else:
                            chi = self.pb.phasevar["chi"]
                        densfuncs = []
                        for n in range(self.pb.num_domains):
                            densfuncs.append(self.pb.vf.get_density(self.pb.rho[n], chi=chi))
                        # dens_proj = project(
                        #     densfuncs,
                        #     self.pb.V_scalar,
                        #     self.pb.dx,
                        #     domids=self.pb.domain_ids,
                        #     nm="Density",
                        #     comm=self.pb.comm,
                        #     entity_maps=self.pb.io.entity_maps,
                        # )
                        dens_out = fem.Function(self.pb.V_out_scalar, name="Density")
                        # dens_out.interpolate(dens_proj)
                        for n, M in enumerate(self.pb.domain_ids):
                            cells_n = self.pb.dx.subdomain_data().find(M)  # TODO: Not working if V_out on submesh! These are parent cells!
                            # dens_out.interpolate(fem.Expression(densfuncs[n], self.pb.V_out_scalar.element.interpolation_points), cells0=cells_n)
                            dens_out.interpolate(fem.Expression(densfuncs[n], self.pb.V_out_scalar.element.interpolation_points))
                        self.pb.resultsfiles[res].write_function(dens_out, indicator)
                    elif res == "chi":  # normalized phase field variable for coefficient evaluation - mainly for testing purposes...
                        if self.pb.io.output_midpoint:
                            chi = self.pb.phasevar["chi_mid"]
                        else:
                            chi = self.pb.phasevar["chi"]
                        chi_out = fem.Function(self.pb.V_out_scalar, name="Chi")
                        chi_out.interpolate(fem.Expression(chi, self.pb.V_out_scalar.element.interpolation_points))
                        self.pb.resultsfiles[res].write_function(chi_out, indicator)
                    elif res == "alpha":
                        afuncs = []
                        for n in range(self.pb.num_domains):
                            afuncs.append(self.pb.alpha[n])
                        a_out = fem.Function(self.pb.V_out_scalar, name="Alpha")
                        for n, M in enumerate(self.pb.domain_ids):
                            cells_n = self.pb.dx.subdomain_data().find(M)  # TODO: Not working if V_out on submesh! These are parent cells!
                            # a_out.interpolate(fem.Expression(afuncs[n], self.pb.V_out_scalar.element.interpolation_points), cells0=cells_n)
                            a_out.interpolate(fem.Expression(afuncs[n], self.pb.V_out_scalar.element.interpolation_points))
                        self.pb.resultsfiles[res].write_function(a_out, indicator)
                    elif res == "internalpower_membrane":
                        pwfuncs = []
                        for n in range(len(self.pb.bintpower)):
                            pwfuncs.append(self.pb.bintpower[n])
                        internalpower_membrane = project(
                            pwfuncs,
                            self.pb.Vd_scalar,
                            self.pb.bmeasures[0],
                            domids=self.pb.idmem,
                            nm="InternalPower_membrane",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        internalpower_membrane_out = fem.Function(self.pb.V_out_scalar, name=internalpower_membrane.name)
                        internalpower_membrane_out.interpolate(internalpower_membrane)
                        self.pb.resultsfiles[res].write_function(internalpower_membrane_out, indicator)
                    elif res == "internalpower":
                        pwfuncs = []
                        for n in range(self.pb.num_domains):
                            pwfuncs.append(
                                ufl.inner(
                                    self.pb.ma[n].sigma(self.pb.v, self.pb.p, F=self.pb.alevar["Fale"], chi=self.pb.phasevar["chi"]),
                                    self.pb.ki.shearrate(self.pb.v, F=self.pb.alevar["Fale"]),
                                )
                            )
                        pw = project(
                            pwfuncs,
                            self.pb.Vd_scalar,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="InternalPower",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        pw_out = fem.Function(self.pb.V_out_scalar, name=pw.name)
                        pw_out.interpolate(pw)
                        self.pb.resultsfiles[res].write_function(pw_out, indicator)
                    else:
                        raise NameError("Unknown output to write for fluid mechanics!")

    def readcheckpoint(self, N_rest):
        vecs_to_read = {}
        vecs_to_read[self.pb.v] = "v"
        vecs_to_read[self.pb.v_old] = "v_old"
        vecs_to_read[self.pb.v_veryold] = "v_veryold" # for BDF2 scheme
        vecs_to_read[self.pb.a_old] = "a_old"
        vecs_to_read[self.pb.uf_old] = "uf_old"  # needed for ALE fluid / FSI / FrSI
        vecs_to_read[self.pb.uf_veryold] = "uf_veryold"  # for BDF2 scheme
        if any(self.pb.mem_active_stress):  # for active membrane model (FrSI)
            vecs_to_read[self.pb.tau_a] = "tau_a"
            vecs_to_read[self.pb.tau_a_old] = "tau_a"

        # pressure may be discontinuous across domains
        if bool(self.pb.io.duplicate_mesh_domains):
            for m, mp in enumerate(self.pb.io.duplicate_mesh_domains):
                vecs_to_read[self.pb.p__[m + 1]] = "p" + str(m + 1)
                vecs_to_read[self.pb.p_old__[m + 1]] = "p_old" + str(m + 1)
        else:
            vecs_to_read[self.pb.p] = "p"
            vecs_to_read[self.pb.p_old] = "p_old"

        for key in vecs_to_read:
            if self.pb.io.restart_io_type == "petscvector":
                # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
                # and for safety reasons, include the number of cores in the dat file name
                viewer = PETSc.Viewer().createMPIIO(
                    self.pb.io.output_path
                    + "/checkpoint_"
                    + self.pb.pbase.simname
                    + "_"
                    + self.pb.problem_physics
                    + "_"
                    + vecs_to_read[key]
                    + "_"
                    + str(N_rest)
                    + "_"
                    + str(self.pb.comm.size)
                    + "proc.dat",
                    "r",
                    self.pb.comm,
                )
                key.x.petsc_vec.load(viewer)
                key.x.petsc_vec.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT,
                    mode=PETSc.ScatterMode.FORWARD,
                )
                viewer.destroy()
            elif self.pb.io.restart_io_type == "plaintext":  # only working for nodal fields!
                self.readfunction(
                    key,
                    self.pb.io.output_path
                    + "/checkpoint_"
                    + self.pb.pbase.simname
                    + "_"
                    + self.pb.problem_physics
                    + "_"
                    + vecs_to_read[key]
                    + "_"
                    + str(N_rest)
                    + ".txt",
                    filetype='plaintext',
                )
            else:
                raise ValueError("Unknown restart_io_type!")

    def writecheckpoint(self, N):
        vecs_to_write = {}
        vecs_to_write[self.pb.v] = "v"
        vecs_to_write[self.pb.v_old] = "v_old"
        vecs_to_write[self.pb.v_veryold] = "v_veryold" # for BDF2 scheme
        vecs_to_write[self.pb.a_old] = "a_old"
        vecs_to_write[self.pb.uf_old] = "uf_old"  # needed for ALE fluid / FSI / FrSI
        vecs_to_write[self.pb.uf_veryold] = "uf_veryold" # for BDF2 scheme
        if any(self.pb.mem_active_stress):
            vecs_to_write[self.pb.tau_a] = "tau_a"

        # pressure may be discontinuous across domains
        if bool(self.pb.io.duplicate_mesh_domains):
            for m, mp in enumerate(self.pb.io.duplicate_mesh_domains):
                vecs_to_write[self.pb.p__[m + 1]] = "p" + str(m + 1)
                vecs_to_write[self.pb.p_old__[m + 1]] = "p_old" + str(m + 1)
        else:
            vecs_to_write[self.pb.p] = "p"
            vecs_to_write[self.pb.p_old] = "p_old"

        for key in vecs_to_write:
            if self.pb.io.restart_io_type == "petscvector":
                # It seems that a vector written by n processors is loaded wrongly by m != n processors! So, we have to restart with the same number of cores,
                # and for safety reasons, include the number of cores in the dat file name
                viewer = PETSc.Viewer().createMPIIO(
                    self.pb.io.output_path
                    + "/checkpoint_"
                    + self.pb.pbase.simname
                    + "_"
                    + self.pb.problem_physics
                    + "_"
                    + vecs_to_write[key]
                    + "_"
                    + str(N)
                    + "_"
                    + str(self.pb.comm.size)
                    + "proc.dat",
                    "w",
                    self.pb.comm,
                )
                key.x.petsc_vec.view(viewer)
                viewer.destroy()
            elif self.pb.io.restart_io_type == "plaintext":  # only working for nodal fields!
                self.writefunction(
                    key,
                    self.pb.io.output_path
                    + "/checkpoint_"
                    + self.pb.pbase.simname
                    + "_"
                    + self.pb.problem_physics
                    + "_"
                    + vecs_to_write[key]
                    + "_"
                    + str(N),
                    filetype='plaintext',
                )
            else:
                raise ValueError("Unknown restart_io_type!")

    def close_output_files(self):
        if self.pb.io.write_results_every > 0:
            for res in self.pb.results_to_write:
                if res not in self.results_pre:
                    if res == "pressure" and bool(self.pb.io.duplicate_mesh_domains):
                        for m, mp in enumerate(self.pb.io.duplicate_mesh_domains):
                            self.pb.resultsfiles[res + str(m + 1)].close()
                    else:
                        self.pb.resultsfiles[res].close()
