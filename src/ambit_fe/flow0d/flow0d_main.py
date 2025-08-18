#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from petsc4py import PETSc
from .. import timeintegration
from ..solver import solver_nonlin
from .. import ioparams
from .. import utilities

from ..base import problem_base, solver_base

# framework of 0D flow models, relating pressure p (and its derivative) to fluxes q


class Flow0DProblem(problem_base):
    def __init__(
        self,
        pbase,
        io_params,
        time_params,
        model_params,
        time_curves,
        coupling_params={},
    ):
        self.pbase = pbase

        # pointer to communicator
        self.comm = self.pbase.comm
        self.comm_sq = self.pbase.comm_sq

        ioparams.check_params_io(io_params)
        ioparams.check_params_time_flow0d(time_params)

        self.problem_physics = "flow0d"

        self.time_params = time_params

        # only relevant to syspul* models
        try:
            self.chamber_models = model_params["chamber_models"]
            if "ao" not in self.chamber_models.keys():
                self.chamber_models["ao"] = {"type": "0D_rigid"}  # add aortic root model
        except:
            self.chamber_models = {}

        self.coronary_model = model_params.get("coronary_model", None)
        self.vad_model = model_params.get("vad_model", None)

        self.excitation_curve = model_params.get("excitation_curve", None)

        initial_file = time_params.get("initial_file", "")

        # could use extra write frequency setting for 0D model (i.e. for coupled problem)
        try:
            self.write_results_every_0D = io_params["write_results_every_0D"]
        except:
            self.write_results_every_0D = io_params["write_results_every"]

        # for restart
        self.write_restart_every = io_params.get("write_restart_every", -1)

        # could use extra output path setting for 0D model (i.e. for coupled problem)
        try:
            self.output_path_0D = io_params["output_path_0D"]
        except:
            self.output_path_0D = io_params["output_path"]

        # whether to output midpoint (t_{n+theta}) of state variables or endpoint (t_{n+1}) - for post-processing
        self.output_midpoint = io_params.get("output_midpoint_0D", False)

        valvelaws = model_params.get(
            "valvelaws",
            {
                "av": ["pwlin_pres", 0],
                "mv": ["pwlin_pres", 0],
                "pv": ["pwlin_pres", 0],
                "tv": ["pwlin_pres", 0],
            },
        )

        self.cq = coupling_params.get("coupling_quantity", ["volume"] * 5)
        self.vq = coupling_params.get("variable_quantity", ["pressure"] * 5)

        self.coup_type = coupling_params.get("coupling_type", None)

        self.eps_periodic = time_params.get("eps_periodic", 1e-20)

        self.periodic_checktype = time_params.get("periodic_checktype", ["allvar"])

        self.prescribed_variables = model_params.get("prescribed_variables", {})

        try:
            self.perturb_type = model_params["perturb_type"][0]
        except:
            self.perturb_type = None
        try:
            self.perturb_factor = model_params["perturb_type"][1]
        except:
            self.perturb_factor = 1.0
        try:
            self.perturb_id = model_params["perturb_type"][2]
        except:
            self.perturb_id = -1

        self.initial_backwardeuler = time_params.get("initial_backwardeuler", False)

        self.ode_parallel = io_params.get("ode_parallel", False)

        self.perturb_after_cylce = model_params.get("perturb_after_cylce", -1)
        # definitely set to -1 if we don't have a perturb type
        if self.perturb_type is None:
            self.perturb_after_cylce = -1

        self.have_induced_pert = False

        # initialize 0D model class - currently, we always init with True since restart will generate new output file names (so no need to append to old ones)
        if model_params["modeltype"] == "2elwindkessel":
            from .cardiovascular0D_2elwindkessel import (
                cardiovascular0D2elwindkessel,
            )

            self.cardvasc0D = cardiovascular0D2elwindkessel(
                model_params["parameters"],
                self.cq,
                self.vq,
                init=True,
                ode_par=self.ode_parallel,
                comm=self.comm,
            )
        elif model_params["modeltype"] == "4elwindkesselLsZ":
            from .cardiovascular0D_4elwindkesselLsZ import (
                cardiovascular0D4elwindkesselLsZ,
            )

            self.cardvasc0D = cardiovascular0D4elwindkesselLsZ(
                model_params["parameters"],
                self.cq,
                self.vq,
                init=True,
                ode_par=self.ode_parallel,
                comm=self.comm,
            )
        elif model_params["modeltype"] == "4elwindkesselLpZ":
            from .cardiovascular0D_4elwindkesselLpZ import (
                cardiovascular0D4elwindkesselLpZ,
            )

            self.cardvasc0D = cardiovascular0D4elwindkesselLpZ(
                model_params["parameters"],
                self.cq,
                self.vq,
                init=True,
                ode_par=self.ode_parallel,
                comm=self.comm,
            )
        elif model_params["modeltype"] == "CRLinoutlink":
            from .cardiovascular0D_CRLinoutlink import (
                cardiovascular0DCRLinoutlink,
            )

            self.cardvasc0D = cardiovascular0DCRLinoutlink(
                model_params["parameters"],
                self.cq,
                self.vq,
                init=True,
                ode_par=self.ode_parallel,
                comm=self.comm,
            )
        elif model_params["modeltype"] == "syspul":
            from .cardiovascular0D_syspul import cardiovascular0Dsyspul

            self.cardvasc0D = cardiovascular0Dsyspul(
                model_params["parameters"],
                self.chamber_models,
                self.cq,
                self.vq,
                valvelaws=valvelaws,
                cormodel=self.coronary_model,
                vadmodel=self.vad_model,
                init=True,
                ode_par=self.ode_parallel,
                comm=self.comm,
            )
        elif model_params["modeltype"] == "syspulcap":
            from .cardiovascular0D_syspulcap import cardiovascular0Dsyspulcap

            self.cardvasc0D = cardiovascular0Dsyspulcap(
                model_params["parameters"],
                self.chamber_models,
                self.cq,
                self.vq,
                valvelaws=valvelaws,
                cormodel=self.coronary_model,
                vadmodel=self.vad_model,
                init=True,
                ode_par=self.ode_parallel,
                comm=self.comm,
            )
        elif model_params["modeltype"] == "syspulcapcor":
            from .cardiovascular0D_syspulcap import (
                cardiovascular0Dsyspulcapcor,
            )

            self.cardvasc0D = cardiovascular0Dsyspulcapcor(
                model_params["parameters"],
                self.chamber_models,
                self.cq,
                self.vq,
                valvelaws=valvelaws,
                cormodel=self.coronary_model,
                vadmodel=self.vad_model,
                init=True,
                ode_par=self.ode_parallel,
                comm=self.comm,
            )
        elif model_params["modeltype"] == "syspulcaprespir":
            from .cardiovascular0D_syspulcaprespir import (
                cardiovascular0Dsyspulcaprespir,
            )

            self.cardvasc0D = cardiovascular0Dsyspulcaprespir(
                model_params["parameters"],
                self.chamber_models,
                self.cq,
                self.vq,
                valvelaws=valvelaws,
                cormodel=self.coronary_model,
                vadmodel=self.vad_model,
                init=True,
                ode_par=self.ode_parallel,
                comm=self.comm,
            )
        else:
            raise NameError("Unknown 0D modeltype!")

        self.numdof = self.cardvasc0D.numdof

        if (
            self.coup_type == "monolithic_direct" and self.comm.size > 1
        ):  # for 3D-0D direct coupling, we need a parallel 0D layout
            assert self.ode_parallel
        if (
            self.coup_type == "monolithic_lagrange" and self.comm.size > 1
        ):  # for 3D-0D constraint-like coupling, we want a serial 0D layout
            assert not self.ode_parallel

        # vectors and matrices
        if self.ode_parallel:
            self.K = PETSc.Mat().createAIJ(
                size=(self.numdof, self.numdof),
                bsize=None,
                nnz=None,
                csr=None,
                comm=self.comm,
            )
        else:
            self.K = PETSc.Mat().create(comm=self.comm_sq)
            self.K.setType(PETSc.Mat.Type.SEQAIJ)
            self.K.setSizes(size=(self.numdof, self.numdof))
        self.K.setUp()

        self.K.assemble()
        self.dK_ = self.K.duplicate(copy=True)
        self.K_ = self.K.duplicate(copy=True)

        self.r = self.K.createVecLeft()

        self.s, self.s_old, self.s_mid = (
            self.K.createVecLeft(),
            self.K.createVecLeft(),
            self.K.createVecLeft(),
        )
        self.sTc, self.sTc_old = self.K.createVecLeft(), self.K.createVecLeft()

        self.df, self.df_old = self.K.createVecLeft(), self.K.createVecLeft()
        self.f, self.f_old = self.K.createVecLeft(), self.K.createVecLeft()

        # tmp vectors for perturbation solves (for LM-coupled 3D-0D problems)
        self.s_tmp, self.df_tmp, self.f_tmp, self.aux_tmp = (
            self.K.createVecLeft(),
            self.K.createVecLeft(),
            self.K.createVecLeft(),
            np.zeros(self.numdof),
        )

        self.aux, self.aux_old, self.aux_mid = (
            np.zeros(self.numdof),
            np.zeros(self.numdof),
            np.zeros(self.numdof),
        )
        self.auxTc, self.auxTc_old = (
            np.zeros(self.numdof),
            np.zeros(self.numdof),
        )

        self.s_set = self.K.createVecLeft()  # set point for multiscale analysis

        self.c, self.y = [], [0] * 4

        self.auxdata, self.auxdata_old = (
            {},
            {},
        )  # auxiliary data that can be set by other fields (e.g. fluxes from the 3D monitor)

        # initialize flow0d time-integration class
        self.ti = timeintegration.timeintegration_flow0d(
            time_params,
            self.pbase.dt,
            self.pbase.numstep,
            time_curves,
            self.pbase.t_init,
            comm=self.comm,
        )

        if initial_file:
            self.initialconditions = self.cardvasc0D.set_initial_from_file(initial_file)
        else:
            self.initialconditions = time_params["initial_conditions"]

        if self.pbase.restart_step == 0:
            self.cardvasc0D.initialize(self.s, self.initialconditions)
            self.cardvasc0D.initialize(self.s_old, self.initialconditions)
            self.cardvasc0D.initialize(self.sTc_old, self.initialconditions)

        self.theta_ost = time_params["theta_ost"]

        # number of fields involved
        self.nfields = 1

        # residual and matrix lists
        self.r_list = [None] * self.nfields
        self.K_list = [[None] * self.nfields for _ in range(self.nfields)]

        self.K_list[0][0] = self.K
        self.r_list[0] = self.r

    def assemble_residual(self, t):
        self.cardvasc0D.evaluate(self.s, t, self.df, self.f, None, None, self.c, self.y, self.aux)

        theta = self.theta0d_timint(t)

        self.df.assemble(), self.df_old.assemble()
        self.f.assemble(), self.f_old.assemble()

        # 0D rhs vector: r = (df - df_old)/dt + theta * f + (1-theta) * f_old
        self.r.zeroEntries()

        self.r.axpy(1.0 / self.pbase.dt, self.df)
        self.r.axpy(-1.0 / self.pbase.dt, self.df_old)

        self.r.axpy(theta, self.f)
        self.r.axpy(1.0 - theta, self.f_old)

        # if we have prescribed variable values over time
        if bool(self.prescribed_variables):
            for a in self.prescribed_variables:
                varindex = self.cardvasc0D.varmap[a]
                prescr = self.prescribed_variables[a]
                prtype = list(prescr.keys())[0]
                if prtype == "val":
                    val = prescr["val"]
                elif prtype == "curve":
                    curvenumber = prescr["curve"]
                    val = self.ti.timecurves(curvenumber)(t)
                elif prtype == "flux_monitor":
                    monid = prescr["flux_monitor"]
                    val = self.auxdata["q"][monid]
                else:
                    raise ValueError("Unknown type to prescribe a variable.")
                self.cardvasc0D.set_prescribed_variables_residual(self.s, self.r, val, varindex)

        self.r_list[0] = self.r

    def assemble_stiffness(self, t):
        self.cardvasc0D.evaluate(self.s, t, None, None, self.dK_, self.K_, self.c, self.y, self.aux)

        theta = self.theta0d_timint(t)

        self.dK_.assemble()
        self.K_.assemble()
        self.K.assemble()

        self.K.zeroEntries()
        self.K.axpy(1.0 / self.pbase.dt, self.dK_)
        self.K.axpy(theta, self.K_)

        # if we have prescribed variable values over time
        if bool(self.prescribed_variables):
            for a in self.prescribed_variables:
                varindex = self.cardvasc0D.varmap[a]
                self.cardvasc0D.set_prescribed_variables_stiffness(self.K, varindex)

        self.K_list[0][0] = self.K

    def theta0d_timint(self, t):
        if self.initial_backwardeuler:
            if np.isclose(t, self.pbase.dt):
                theta = 1.0
            else:
                theta = self.theta_ost
        else:
            theta = self.theta_ost

        return theta

    def writerestart(self, sname, N, ms=False):
        self.cardvasc0D.write_restart(self.output_path_0D, sname + "_s", N, self.s)
        self.cardvasc0D.write_restart(self.output_path_0D, sname + "_aux", N, self.aux)
        self.cardvasc0D.write_restart(self.output_path_0D, sname + "_sTc_old", N, self.sTc_old)
        if ms:
            self.cardvasc0D.write_restart(self.output_path_0D, sname + "_s_set", N, self.s_set)

        if self.cardvasc0D.T_cycl > 0:  # write heart cycle info
            if self.comm.rank == 0:
                f = open(
                    self.output_path_0D + "/checkpoint_" + sname + "_cycledata_" + str(N) + ".txt",
                    "wt",
                )
                f.write("%i %.8f" % (self.ti.cycle[0], self.ti.cycleerror[0]))
                f.close()

        # write auxdata
        if bool(self.auxdata_old):
            if bool(self.auxdata_old["p"]):
                if self.comm.rank == 0:
                    f = open(
                        self.output_path_0D + "/checkpoint_" + sname + "_auxdata_old_p_" + str(N) + ".txt",
                        "wt",
                    )
                    for m in self.auxdata_old["p"].keys():
                        f.write("%.16E\n" % (self.auxdata_old["p"][m]))
                    f.close()

    def readrestart(self, sname, rst, ms=False):
        self.cardvasc0D.read_restart(self.output_path_0D, sname + "_s", rst, self.s)
        self.cardvasc0D.read_restart(self.output_path_0D, sname + "_s", rst, self.s_old)
        self.cardvasc0D.read_restart(self.output_path_0D, sname + "_aux", rst, self.aux)
        self.cardvasc0D.read_restart(self.output_path_0D, sname + "_aux", rst, self.aux_old)
        self.cardvasc0D.read_restart(self.output_path_0D, sname + "_sTc_old", rst, self.sTc_old)
        if ms:
            self.cardvasc0D.read_restart(self.output_path_0D, sname + "_s_set", rst, self.s_set)

        if self.cardvasc0D.T_cycl > 0:  # read heart cycle info
            self.ti.cycle[0] = np.loadtxt(
                self.output_path_0D + "/checkpoint_" + sname + "_cycledata_" + str(rst) + ".txt",
                usecols=(0),
                dtype=int,
            )
            self.ti.cycleerror[0] = np.loadtxt(
                self.output_path_0D + "/checkpoint_" + sname + "_cycledata_" + str(rst) + ".txt",
                usecols=(1),
                dtype=float,
            )
            self.pbase.t_init -= (self.ti.cycle[0] - 1) * self.cardvasc0D.T_cycl

        if bool(self.auxdata_old):
            if bool(self.auxdata_old["p"]):
                auxdata_p = np.loadtxt(
                    self.output_path_0D + "/checkpoint_" + sname + "_auxdata_old_p_" + str(rst) + ".txt",
                    ndmin=1,
                )
                for m in range(len(auxdata_p)):
                    self.auxdata_old["p"][m] = auxdata_p[m]

    def evaluate_activation(self, t):
        # activation curves
        if bool(self.chamber_models):
            ci = 0
            for i, ch in enumerate(["lv", "rv", "la", "ra"]):
                if self.chamber_models[ch]["type"] == "0D_elast":
                    self.y[i] = self.ti.timecurves(self.chamber_models[ch]["activation_curve"])(t)
                    ci += 1
                if self.chamber_models[ch]["type"] == "0D_elast_prescr":
                    self.y[i] = self.ti.timecurves(self.chamber_models[ch]["elastance_curve"])(t)
                    ci += 1
                if self.chamber_models[ch]["type"] == "0D_prescr":
                    self.c[self.len_c_3d0d + ci] = self.ti.timecurves(self.chamber_models[ch]["prescribed_curve"])(t)
                    ci += 1

    def induce_perturbation(self):
        if self.perturb_after_cylce > 0:  # at least run through one healthy cycle
            if self.ti.cycle[0] > self.perturb_after_cylce:
                utilities.print_status(
                    ">>> Induced cardiovascular disease type: %s" % (self.perturb_type),
                    self.comm,
                )

                self.cardvasc0D.induce_perturbation(self.perturb_type, self.perturb_factor)
                self.have_induced_pert = True

    ### now the base routines for this problem

    def read_restart(self, sname, N):
        # read restart information
        if self.pbase.restart_step > 0:
            self.readrestart(sname + "_" + self.problem_physics, N)

    def evaluate_initial(self):
        # evaluate old state
        if self.excitation_curve is not None:
            for i in range(len(self.excitation_curve)):
                self.c.append(self.ti.timecurves(self.excitation_curve[i])(self.pbase.t_init))
        if bool(self.chamber_models):
            for i, ch in enumerate(["lv", "rv", "la", "ra"]):
                if self.chamber_models[ch]["type"] == "0D_elast":
                    self.y[i] = self.ti.timecurves(self.chamber_models[ch]["activation_curve"])(self.pbase.t_init)
                if self.chamber_models[ch]["type"] == "0D_elast_prescr":
                    self.y[i] = self.ti.timecurves(self.chamber_models[ch]["elastance_curve"])(self.pbase.t_init)
                if self.chamber_models[ch]["type"] == "0D_prescr":
                    self.c.append(self.ti.timecurves(self.chamber_models[ch]["prescribed_curve"])(self.pbase.t_init))

        # if we have prescribed variable values over time
        if self.pbase.restart_step == 0:  # we read s and s_old in case of restart
            if bool(self.prescribed_variables):
                for a in self.prescribed_variables:
                    varindex = self.cardvasc0D.varmap[a]
                    prescr = self.prescribed_variables[a]
                    prtype = list(prescr.keys())[0]
                    if prtype == "val":
                        val = prescr["val"]
                    elif prtype == "curve":
                        curvenumber = prescr["curve"]
                        val = self.ti.timecurves(curvenumber)(self.pbase.t_init)
                    else:
                        raise ValueError("Unknown type to prescribe a variable.")
                    self.s[varindex], self.s_old[varindex] = val, val

        self.cardvasc0D.evaluate(
            self.s_old,
            self.pbase.t_init,
            self.df_old,
            self.f_old,
            None,
            None,
            self.c,
            self.y,
            self.aux_old,
        )
        self.auxTc_old[:] = self.aux_old[:]

    def write_output_ini(self):
        pass

    def write_output_pre(self):
        pass

    def evaluate_pre_solve(self, t, N, dt):
        # external volume/flux from time curve
        if self.excitation_curve is not None:
            for i in range(len(self.excitation_curve)):
                self.c[i] = self.ti.timecurves(self.excitation_curve[i])(t)
        # activation curves
        self.evaluate_activation(t)

    def evaluate_post_solve(self, t, N):
        pass

    def set_output_state(self, t):
        # get midpoint dof values for post-processing (has to be called before update!)
        self.s.assemble(), self.s_old.assemble(), self.s_mid.assemble()
        self.cardvasc0D.set_output_state(
            self.s,
            self.s_old,
            self.s_mid,
            self.theta0d_timint(t),
            midpoint=self.output_midpoint,
        )
        self.cardvasc0D.set_output_state(
            self.aux,
            self.aux_old,
            self.aux_mid,
            self.theta0d_timint(t),
            midpoint=self.output_midpoint,
        )

    def write_output(self, N, t):
        # raw txt file output of 0D model quantities
        if self.write_results_every_0D > 0 and N % self.write_results_every_0D == 0:
            self.cardvasc0D.write_output(
                self.output_path_0D,
                t,
                self.s_mid,
                self.aux_mid,
                self.pbase.simname + "_" + self.problem_physics,
            )

    def update(self):
        # update timestep
        self.cardvasc0D.update(
            self.s,
            self.df,
            self.f,
            self.s_old,
            self.df_old,
            self.f_old,
            self.aux,
            self.aux_old,
        )

    def print_to_screen(self):
        self.s_mid.assemble()
        self.cardvasc0D.print_to_screen(self.s_mid, self.aux_mid)

    def induce_state_change(self):
        # induce some disease/perturbation for cardiac cycle (i.e. valve stenosis or leakage)
        if self.perturb_type is not None and not self.have_induced_pert:
            self.induce_perturbation()

    def write_restart(self, sname, N, force=False):
        # write 0D restart info - old and new quantities are the same at this stage (except cycle values sTc)
        if (self.write_restart_every > 0 and N % self.write_restart_every == 0) or force:
            self.writerestart(sname + "_" + self.problem_physics, N)

        if bool(self.auxdata_old):
            if bool(self.auxdata_old["p"]):
                # auxdata update - needs to be done here since timestep update is called prior to write_restart,
                # but we need the values from the pre-previous step to get the correct restart (pressure auxdata)
                # is only evaluated after a solve, hence technically within a step it's always "old" data
                for k in self.auxdata["p"]:
                    self.auxdata_old["p"][k] = self.auxdata["p"][k]

    def check_abort(self, t):
        # check for periodicity in cardiac cycle and stop if reached (only for syspul* models - cycle counter gets updated here)
        is_periodic = self.cardvasc0D.cycle_check(
            self.s,
            self.sTc,
            self.sTc_old,
            self.aux,
            self.auxTc,
            self.auxTc_old,
            t,
            self.ti.cycle,
            self.ti.cycleerror,
            self.eps_periodic,
            check=self.periodic_checktype,
            inioutpath=self.output_path_0D,
            nm=self.pbase.simname,
            induce_pert_after_cycl=self.perturb_after_cylce,
        )

        if is_periodic:
            utilities.print_status(
                "Periodicity reached after %i heart cycles with cycle error %.4f! Finished. :-)"
                % (self.ti.cycle[0] - 1, self.ti.cycleerror[0]),
                self.comm,
            )

        return is_periodic

    def destroy(self):
        self.dK_.destroy()
        self.K_.destroy()


class Flow0DSolver(solver_base):
    def initialize_nonlinear_solver(self):
        self.evaluate_system_initial()

        # initialize nonlinear solver class
        self.solnln = solver_nonlin.solver_nonlinear_ode([self.pb], self.solver_params)

    def solve_initial_state(self):
        pass

    def solve_nonlinear_problem(self, t):
        self.solnln.newton(t)

    def print_timestep_info(self, N, t, ni, li, wt):
        # print time step info to screen
        self.pb.ti.print_timestep(N, t, self.solnln.lsp, self.pb.pbase.numstep, ni=ni, li=li, wt=wt)
