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


class IO_ale(IO_field):
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
                    if res == "aledisplacement":
                        d_out = fem.Function(self.pb.V_out_vector, name=self.pb.d.name)
                        d_out.interpolate(self.pb.d)
                        self.pb.resultsfiles[res].write_function(d_out, indicator)
                    elif res == "alevelocity":
                        w_proj = project(
                            self.pb.wel,
                            self.pb.V_d,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="AleVelocity",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        w_out = fem.Function(self.pb.V_out_vector, name=w_proj.name)
                        w_out.interpolate(w_proj)
                        self.pb.resultsfiles[res].write_function(w_out, indicator)
                    elif res == "alestress": # might be needed for some debugging purposes...
                        stressfuncs = []
                        for n in range(self.pb.num_domains):
                            stressfuncs.append(self.pb.ma[n].stress(self.pb.d, self.pb.wel))
                        stress = project(
                            stressfuncs,
                            self.pb.Vd_tensor,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="AleStress",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        stress_out = fem.Function(self.pb.V_out_tensor, name=stress.name)
                        stress_out.interpolate(stress)
                        self.pb.resultsfiles[res].write_function(stress_out, indicator)
                    else:
                        raise NameError("Unknown output to write for ALE mechanics!")

    def readcheckpoint(self, N_rest):
        vecs_to_read = {}
        vecs_to_read[self.pb.d] = "d"
        vecs_to_read[self.pb.d_old] = "d_old"
        vecs_to_read[self.pb.d_veryold] = "d_veryold"
        vecs_to_read[self.pb.w_old] = "w_old"

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
        vecs_to_write[self.pb.d] = "d"
        vecs_to_write[self.pb.d_old] = "d_old"
        vecs_to_write[self.pb.d_veryold] = "d_veryold"
        vecs_to_write[self.pb.w_old] = "w_old"

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
