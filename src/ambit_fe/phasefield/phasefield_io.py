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
                        if self.pb.io.output_midpoint:
                            phi_proj = project(
                                self.pb.phi_mid,
                                self.pb.V_phi,
                                self.pb.dx,
                                domids=self.pb.domain_ids,
                                nm="PhaseField",
                                comm=self.pb.comm,
                                entity_maps=self.pb.io.entity_maps,
                            )
                            phi_out = fem.Function(self.pb.V_out_scalar, name=phi_proj.name)
                            phi_out.interpolate(phi_proj)
                        else:
                            phi_out = fem.Function(self.pb.V_out_scalar, name=self.pb.phi.name)
                            phi_out.interpolate(self.pb.phi)
                        self.pb.resultsfiles[res].write_function(phi_out, indicator)
                    elif res == "potential":
                        if self.pb.io.output_midpoint:
                            mu_proj = project(
                                self.pb.mu_mid,
                                self.pb.V_mu,
                                self.pb.dx,
                                domids=self.pb.domain_ids,
                                nm="Potential",
                                comm=self.pb.comm,
                                entity_maps=self.pb.io.entity_maps,
                            )
                            mu_out = fem.Function(self.pb.V_out_scalar, name=mu_proj.name)
                            mu_out.interpolate(mu_proj)
                        else:
                            mu_out = fem.Function(self.pb.V_out_scalar, name=self.pb.mu.name)
                            mu_out.interpolate(self.pb.mu)
                        self.pb.resultsfiles[res].write_function(mu_out, indicator)
                    else:
                        raise NameError("Unknown output to write for Cahn-Hilliard problem!")

    def readcheckpoint(self, N_rest):
        vecs_to_read = {}
        vecs_to_read[self.pb.phi] = "phi"
        vecs_to_read[self.pb.phi_old] = "phi_old"
        vecs_to_read[self.pb.phi_veryold] = "phi_veryold" # for BDF2 scheme
        vecs_to_read[self.pb.phidot_old] = "phidot_old"
        vecs_to_read[self.pb.mu] = "mu"
        vecs_to_read[self.pb.mu_old] = "mu_old"

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
        vecs_to_write[self.pb.phi] = "phi"
        vecs_to_write[self.pb.phi_old] = "phi_old"
        vecs_to_write[self.pb.phi_veryold] = "phi_veryold"
        vecs_to_write[self.pb.phidot_old] = "phidot_old"
        vecs_to_write[self.pb.mu] = "mu"
        vecs_to_write[self.pb.mu_old] = "mu_old"

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
