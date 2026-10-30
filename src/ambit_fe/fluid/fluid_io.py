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
        self.results_pre = ["fibers", "dissipation", "kinetic_energy", "energy_bodyforce", "internal_power", "strain_energy_membrane", "internal_power_membrane"]

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
                        v_out = fem.Function(self.pb.V_out_vector, name=self.pb.v.name)
                        if self.pb.io.output_midpoint:
                            v_out.interpolate(fem.Expression(self.pb.vel_mid, self.pb.V_out_vector.element.interpolation_points))
                        else:
                            v_out.interpolate(self.pb.v)
                        self.pb.resultsfiles[res].write_function(v_out, indicator)
                    elif res == "acceleration":
                        if self.pb.io.output_midpoint:
                            acc = self.pb.acc_mid
                        else:
                            acc = self.pb.acc
                        a_out = fem.Function(self.pb.V_out_vector, name="Acceleration")
                        a_out.interpolate(fem.Expression(acc, self.pb.V_out_vector.element.interpolation_points))
                        self.pb.resultsfiles[res].write_function(a_out, indicator)
                    elif res == "pressure":
                        if bool(self.pb.io.duplicate_mesh_domains):
                            for m, mp in enumerate(self.pb.io.duplicate_mesh_domains):
                                V_out_scalar_sub = fem.functionspace(
                                    self.pb.io.submshes_emap[m + 1][0],
                                    ("Lagrange", self.pb.mesh_degree),
                                )
                                p_out = fem.Function(V_out_scalar_sub, name=self.pb.p_[m].name)
                                if self.pb.io.output_midpoint:
                                    p_out.interpolate(fem.Expression(self.pb.pf_mid_[m], V_out_scalar_sub.element.interpolation_points))
                                else:
                                    p_out.interpolate(self.pb.p_[m])
                                self.pb.resultsfiles[res + str(m + 1)].write_function(p_out, indicator)
                        else:
                            p_out = fem.Function(self.pb.V_out_scalar, name=self.pb.p_[0].name)
                            if self.pb.io.output_midpoint:
                                p_out.interpolate(fem.Expression(self.pb.pf_mid_[0], self.pb.V_out_scalar.element.interpolation_points))
                            else:
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
                    elif res == "fluiddisplacement":
                        if self.pb.io.output_midpoint:
                            uf = self.pb.ufluid_mid
                        else:
                            uf = self.pb.ufluid
                        uf_out = fem.Function(self.pb.V_out_vector, name="FluidDisplacement")
                        uf_out.interpolate(fem.Expression(uf, self.pb.V_out_vector.element.interpolation_points))
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
                    elif res == "strain_energy_membrane_local":
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
                    elif res == "tau_m":  # stabilization parameter tau_m
                        if self.pb.io.output_midpoint:
                            tau = self.pb.tau_m_mid
                        else:
                            tau = self.pb.tau_m
                        tau_out = fem.Function(self.pb.V_out_scalar, name="tau_m")
                        tau_out.interpolate(fem.Expression(tau, self.pb.V_out_scalar.element.interpolation_points))  # TODO: Not working if V_out on submesh! Why?
                        self.pb.resultsfiles[res].write_function(tau_out, indicator)
                    elif res == "tau_c":  # stabilization parameter tau_c
                        if self.pb.io.output_midpoint:
                            tau = self.pb.tau_c_mid
                        else:
                            tau = self.pb.tau_c
                        tau_out = fem.Function(self.pb.V_out_scalar, name="tau_c")
                        tau_out.interpolate(fem.Expression(tau, self.pb.V_out_scalar.element.interpolation_points))  # TODO: Not working if V_out on submesh! Why?
                        self.pb.resultsfiles[res].write_function(tau_out, indicator)
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
                    elif res == "internal_power_membrane_local":
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
                    elif res == "internal_power_local":
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
                    elif res == "dissipation":
                        # only global output supported, not as field
                        pass
                    elif res == "kinetic_energy":
                        # only global output supported, not as field
                        pass
                    elif res == "energy_bodyforce":
                        # only global output supported, not as field
                        pass
                    elif res == "internal_power":
                        # global output - use internal_power_local for distributed version
                        pass
                    elif res == "strain_energy_membrane":
                        # global output - use strain_energy_membrane_local for distributed version
                        pass
                    elif res == "internal_power_membrane":
                        # global output - use internal_power_membrane_local for distributed version
                        pass
                    else:
                        raise NameError("Unknown output to write for fluid mechanics!")

    def readcheckpoint(self, N_rest):
        vecs_to_read = {}
        vecs_to_read["v"] = self.pb.v.x.petsc_vec
        vecs_to_read["v_old"] = self.pb.v_old.x.petsc_vec
        vecs_to_read["v_veryold"] = self.pb.v_veryold.x.petsc_vec  # for BDF2 scheme
        if self.pb.ti.discretely_conservative:
            for n in range(self.pb.num_domains):
                vecs_to_read["amom_old" + str(n+1)] = self.pb.amom_old[n].x.petsc_vec
        else:
            vecs_to_read["a_old"] = self.pb.a_old.x.petsc_vec
        vecs_to_read["uf_old"] = self.pb.uf_old.x.petsc_vec  # needed for ALE fluid / FSI / FrSI
        vecs_to_read["uf_veryold"] = self.pb.uf_veryold.x.petsc_vec  # for BDF2 scheme
        if any(self.pb.mem_active_stress):  # for active membrane model (FrSI)
            vecs_to_read["tau_a"] = self.pb.tau_a.x.petsc_vec
            vecs_to_read["tau_a_old"] = self.pb.tau_a_old.x.petsc_vec

        # pressure may be discontinuous across domains
        if bool(self.pb.io.duplicate_mesh_domains):
            for m, mp in enumerate(self.pb.io.duplicate_mesh_domains):
                vecs_to_read["p" + str(m + 1)] = self.pb.p__[m + 1].x.petsc_vec
                vecs_to_read["p_old" + str(m + 1)] = self.pb.p_old__[m + 1].x.petsc_vec
        else:
            vecs_to_read["p"] = self.pb.p.x.petsc_vec
            vecs_to_read["p_old"] = self.pb.p_old.x.petsc_vec

        for key, value in vecs_to_read.items():
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
                    + key
                    + "_"
                    + str(N_rest)
                    + "_"
                    + str(self.pb.comm.size)
                    + "proc.dat",
                    "r",
                    self.pb.comm,
                )
                value.load(viewer)
                value.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT,
                    mode=PETSc.ScatterMode.FORWARD,
                )
                viewer.destroy()
            elif self.pb.io.restart_io_type == "plaintext":  # only working for nodal fields!
                raise RuntimeError("Plain text checkpointing currently disabled! Do we really need it?!")
                self.readfunction(
                    value,
                    self.pb.io.output_path
                    + "/checkpoint_"
                    + self.pb.pbase.simname
                    + "_"
                    + self.pb.problem_physics
                    + "_"
                    + key
                    + "_"
                    + str(N_rest)
                    + ".txt",
                    filetype="plaintext",
                )
            else:
                raise ValueError("Unknown restart_io_type!")

    def writecheckpoint(self, N):
        vecs_to_write = {}
        vecs_to_write["v"] = self.pb.v.x.petsc_vec
        vecs_to_write["v_old"] = self.pb.v_old.x.petsc_vec
        vecs_to_write["v_veryold"] = self.pb.v_veryold.x.petsc_vec  # for BDF2 scheme
        if self.pb.ti.discretely_conservative:
            for n in range(self.pb.num_domains):
                vecs_to_write["amom_old" + str(n+1)] = self.pb.amom_old[n].x.petsc_vec
        else:
            vecs_to_write["a_old"] = self.pb.a_old.x.petsc_vec
        vecs_to_write["uf_old"] = self.pb.uf_old.x.petsc_vec  # needed for ALE fluid / FSI / FrSI
        vecs_to_write["uf_veryold"] = self.pb.uf_veryold.x.petsc_vec  # for BDF2 scheme
        if any(self.pb.mem_active_stress):
            vecs_to_write["tau_a"] = self.pb.tau_a.x.petsc_vec
            vecs_to_write["tau_a_old"] = self.pb.tau_a_old.x.petsc_vec

        # pressure may be discontinuous across domains
        if bool(self.pb.io.duplicate_mesh_domains):
            for m, mp in enumerate(self.pb.io.duplicate_mesh_domains):
                vecs_to_write["p" + str(m + 1)] = self.pb.p__[m + 1].x.petsc_vec
                vecs_to_write["p_old" + str(m + 1)] = self.pb.p_old__[m + 1].x.petsc_vec
        else:
            vecs_to_write["p"] = self.pb.p.x.petsc_vec
            vecs_to_write["p_old"] = self.pb.p_old.x.petsc_vec

        for key, value in vecs_to_write.items():
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
                    + key
                    + "_"
                    + str(N)
                    + "_"
                    + str(self.pb.comm.size)
                    + "proc.dat",
                    "w",
                    self.pb.comm,
                )
                value.view(viewer)
                viewer.destroy()
            elif self.pb.io.restart_io_type == "plaintext":  # only working for nodal fields!
                raise RuntimeError("Plain text checkpointing currently disabled! Do we really need it?!")
                self.writefunction(
                    value,
                    self.pb.io.output_path
                    + "/checkpoint_"
                    + self.pb.pbase.simname
                    + "_"
                    + self.pb.problem_physics
                    + "_"
                    + key
                    + "_"
                    + str(N),
                    filetype="plaintext",
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
