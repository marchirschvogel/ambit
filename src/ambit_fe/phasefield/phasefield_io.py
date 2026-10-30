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


class IO_phasefield(IO_field):
    def __init__(self, pb):
        self.pb = pb
        self.results_pre = ["dissipation", "energy", "phasevolumes"]

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
                    if res == "phase":
                        phi_out = fem.Function(self.pb.V_out_scalar, name=self.pb.phi.name)
                        if self.pb.io.output_midpoint:
                            phi_out.interpolate(fem.Expression(self.pb.phi_mid, self.pb.V_out_scalar.element.interpolation_points))
                        else:
                            phi_out.interpolate(self.pb.phi)
                        self.pb.resultsfiles[res].write_function(phi_out, indicator)
                    elif res == "potential":
                        mu_out = fem.Function(self.pb.V_out_scalar, name=self.pb.mu.name)
                        if self.pb.io.output_midpoint:
                            mu_out.interpolate(fem.Expression(self.pb.mu_mid, self.pb.V_out_scalar.element.interpolation_points))
                        else:
                            mu_out.interpolate(self.pb.mu)
                        self.pb.resultsfiles[res].write_function(mu_out, indicator)
                    elif res == "diffusive_flux":
                        fluxfuncs = []
                        for n in range(self.pb.num_domains):
                            if self.pb.io.output_midpoint:
                                mu, phi, p, F = self.pb.mu_mid, self.pb.phi_mid, self.pb.fluidvar["p_mid"], self.pb.alevar["Fale_mid"]
                            else:
                                mu, phi, p, F = self.pb.mu, self.pb.phi, self.pb.fluidvar["p"], self.pb.alevar["Fale"]
                            fluxfuncs.append(self.pb.ma[n].diffusive_flux(mu, phi, p=p, F=F, alpha=self.pb.fluidvar["alpha"][n]))
                        diff_flux = project(
                            fluxfuncs,
                            self.pb.Vd_mu_vector,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="DiffusiveFlux",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        diff_flux_out = fem.Function(self.pb.V_out_vector, name=diff_flux.name)
                        diff_flux_out.interpolate(diff_flux)
                        self.pb.resultsfiles[res].write_function(diff_flux_out, indicator)
                    elif res == "dissipation":
                        # only global output supported, not as field
                        pass
                    elif res == "energy":
                        # only global output supported, not as field
                        pass
                    elif res == "phasevolumes":
                        # only global output
                        pass
                    else:
                        raise NameError("Unknown output to write for Cahn-Hilliard problem!")

    def readcheckpoint(self, N_rest):
        vecs_to_read = {}
        vecs_to_read["phi"] = self.pb.phi.x.petsc_vec
        vecs_to_read["phi_old"] = self.pb.phi_old.x.petsc_vec
        vecs_to_read["phi_veryold"] = self.pb.phi_veryold.x.petsc_vec  # for BDF2 scheme
        vecs_to_read["phidot_old"] = self.pb.phidot_old.x.petsc_vec
        vecs_to_read["mu"] = self.pb.mu.x.petsc_vec
        vecs_to_read["mu_old"] = self.pb.mu_old.x.petsc_vec

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
        vecs_to_write["phi"] = self.pb.phi.x.petsc_vec
        vecs_to_write["phi_old"] = self.pb.phi_old.x.petsc_vec
        vecs_to_write["phi_veryold"] = self.pb.phi_veryold.x.petsc_vec
        vecs_to_write["phidot_old"] = self.pb.phidot_old.x.petsc_vec
        vecs_to_write["mu"] = self.pb.mu.x.petsc_vec
        vecs_to_write["mu_old"] = self.pb.mu_old.x.petsc_vec

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
