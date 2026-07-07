#!/usr/bin/env python3

# Copyright (c) 2019-2026, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

from dolfinx import fem, io
from petsc4py import PETSc
import ufl
from ..solver.projection import project
from ..mathutils import spectral_decomposition_3x3
from ..ioroutines import IO_field


class IO_solid(IO_field):
    def __init__(self, pb):
        self.pb = pb
        self.results_pre = ["fibers"]

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
            raise ValueError("Unknown indicate_results_by option. Choose 'time' or 'step'.")

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
                        outfile.write_mesh(self.pb.io.mesh)
                        self.pb.resultsfiles[res] = outfile

            return

        else:
            # write results every write_results_every steps
            if self.pb.io.write_results_every > 0 and N % self.pb.io.write_results_every == 0:
                # save solution to XDMF format
                for res in self.pb.results_to_write:
                    if res == "displacement":
                        u_out = fem.Function(self.pb.V_out_vector, name=self.pb.u.name)
                        if self.pb.io.output_midpoint:
                            u_out.interpolate(fem.Expression(self.pb.us_mid, self.pb.V_out_vector.element.interpolation_points))
                        else:
                            u_out.interpolate(self.pb.u)
                        self.pb.resultsfiles[res].write_function(u_out, indicator)
                    elif res == "velocity":
                        if self.pb.io.output_midpoint:
                            vel = self.pb.vel_mid
                        else:
                            vel = self.pb.vel
                        self.v_out = fem.Function(self.pb.V_out_vector, name="Velocity")  # class variable for testing
                        self.v_out.interpolate(fem.Expression(vel, self.pb.V_out_vector.element.interpolation_points))
                        self.pb.resultsfiles[res].write_function(self.v_out, indicator)
                    elif res == "acceleration":
                        if self.pb.io.output_midpoint:
                            acc = self.pb.acc_mid
                        else:
                            acc = self.pb.acc
                        self.a_out = fem.Function(self.pb.V_out_vector, name="Acceleration")  # class variable for testing
                        self.a_out.interpolate(fem.Expression(acc, self.pb.V_out_vector.element.interpolation_points))
                        self.pb.resultsfiles[res].write_function(self.a_out, indicator)
                    elif res == "pressure":
                        if self.pb.p is not None:
                            p_out = fem.Function(self.pb.V_out_scalar, name=self.pb.p.name)
                            if self.pb.io.output_midpoint:
                                p_out.interpolate(fem.Expression(self.pb.ps_mid, self.pb.V_out_scalar.element.interpolation_points))
                            else:
                                p_out.interpolate(self.pb.p)
                            self.pb.resultsfiles[res].write_function(p_out, indicator)
                    elif res == "porepressure":
                        if self.pb.pporo is not None:
                            pp_out = fem.Function(self.pb.V_out_scalar, name=self.pb.pporo.name)
                            if self.pb.io.output_midpoint:
                                pp_out.interpolate(fem.Expression(self.pb.pporo_mid, self.pb.V_out_scalar.element.interpolation_points))
                            else:
                                pp_out.interpolate(self.pb.pporo)
                            self.pb.resultsfiles[res].write_function(pp_out, indicator)
                    elif res == "cauchystress":
                        if self.pb.io.output_midpoint:
                            u, p, v, ivars = self.pb.us_mid, self.pb.ps_mid, self.pb.vel_mid, self.pb.internalvars_mid
                        else:
                            u, p, v, ivars = self.pb.u, self.pb.p, self.pb.vel, self.pb.internalvars
                        stressfuncs = []
                        for n in range(self.pb.num_domains):
                            stressfuncs.append(self.pb.ma[n].sigma(u, p, v, ivar=ivars))
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
                    elif res == "cauchystress_nodal":
                        if self.pb.io.output_midpoint:
                            u, p, v, ivars = self.pb.us_mid, self.pb.ps_mid, self.pb.vel_mid, self.pb.internalvars_mid
                        else:
                            u, p, v, ivars = self.pb.u, self.pb.p, self.pb.vel, self.pb.internalvars
                        stressfuncs = []
                        for n in range(self.pb.num_domains):
                            stressfuncs.append(self.pb.ma[n].sigma(u, p, v, ivar=ivars))
                        cauchystress_nodal = project(
                            stressfuncs,
                            self.pb.V_tensor,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="CauchyStress_nodal",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        cauchystress_nodal_out = fem.Function(self.pb.V_out_tensor, name=cauchystress_nodal.name)
                        cauchystress_nodal_out.interpolate(cauchystress_nodal)
                        self.pb.resultsfiles[res].write_function(cauchystress_nodal_out, indicator)
                    elif res == "cauchystress_principal":
                        if self.pb.io.output_midpoint:
                            u, p, v, ivars = self.pb.us_mid, self.pb.ps_mid, self.pb.vel_mid, self.pb.internalvars_mid
                        else:
                            u, p, v, ivars = self.pb.u, self.pb.p, self.pb.vel, self.pb.internalvars
                        stressfuncs_eval = []
                        for n in range(self.pb.num_domains):
                            evals, _, _ = spectral_decomposition_3x3(
                                self.pb.ma[n].sigma(u, p, v, ivar=ivars)
                            )
                            stressfuncs_eval.append(ufl.as_vector(evals))  # written as vector
                        cauchystress_principal = project(
                            stressfuncs_eval,
                            self.pb.Vd_vector,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="CauchyStress_princ",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        cauchystress_principal_out = fem.Function(self.pb.V_out_vector, name=cauchystress_principal.name)
                        cauchystress_principal_out.interpolate(cauchystress_principal)
                        self.pb.resultsfiles[res].write_function(cauchystress_principal_out, indicator)
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
                    elif res == "cauchystress_membrane_principal":
                        stressfuncs = []
                        for n in range(len(self.pb.bstress)):
                            evals, _, _ = spectral_decomposition_3x3(self.pb.bstress[n])
                            stressfuncs.append(ufl.as_vector(evals))  # written as vector
                        self.cauchystress_membrane_principal = project(
                            stressfuncs,
                            self.pb.Vd_vector,
                            self.pb.bmeasures[0],
                            domids=self.pb.idmem,
                            nm="CauchyStress_membrane_princ",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        cauchystress_membrane_principal_out = fem.Function(
                            self.pb.V_out_vector,
                            name=self.cauchystress_membrane_principal.name,
                        )
                        cauchystress_membrane_principal_out.interpolate(self.cauchystress_membrane_principal)
                        self.pb.resultsfiles[res].write_function(cauchystress_membrane_principal_out, indicator)
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
                    elif res == "trmandelstress":
                        if self.pb.io.output_midpoint:
                            u, p, v, ivars = self.pb.us_mid, self.pb.ps_mid, self.pb.vel_mid, self.pb.internalvars_mid
                        else:
                            u, p, v, ivars = self.pb.u, self.pb.p, self.pb.vel, self.pb.internalvars
                        stressfuncs = []
                        for n in range(self.pb.num_domains):
                            stressfuncs.append(
                                ufl.tr(
                                    self.pb.ma[n].M(
                                        u,
                                        p,
                                        v,
                                        ivar=ivars,
                                    )
                                )
                            )
                        trmandelstress = project(
                            stressfuncs,
                            self.pb.Vd_scalar,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="trMandelStress",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        trmandelstress_out = fem.Function(self.pb.V_out_scalar, name=trmandelstress.name)
                        trmandelstress_out.interpolate(trmandelstress)
                        self.pb.resultsfiles[res].write_function(trmandelstress_out, indicator)
                    elif res == "trmandelstress_e":
                        if self.pb.io.output_midpoint:
                            u, p, v, ivars = self.pb.us_mid, self.pb.ps_mid, self.pb.vel_mid, self.pb.internalvars_mid
                        else:
                            u, p, v, ivars = self.pb.u, self.pb.p, self.pb.vel, self.pb.internalvars
                        stressfuncs = []
                        for n in range(self.pb.num_domains):
                            if self.pb.mat_growth[n]:
                                stressfuncs.append(
                                    ufl.tr(
                                        self.pb.ma[n].M_e(
                                            u,
                                            p,
                                            v,
                                            self.pb.ki.C(u),
                                            ivar=ivars,
                                        )
                                    )
                                )
                            else:
                                stressfuncs.append(ufl.as_ufl(0))
                        trmandelstress_e = project(
                            stressfuncs,
                            self.pb.Vd_scalar,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="trMandelStress_e",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        trmandelstress_e_out = fem.Function(self.pb.V_out_scalar, name=trmandelstress_e.name)
                        trmandelstress_e_out.interpolate(trmandelstress_e)
                        self.pb.resultsfiles[res].write_function(trmandelstress_e_out, indicator)
                    elif res == "vonmises_cauchystress":
                        if self.pb.io.output_midpoint:
                            u, p, v, ivars = self.pb.us_mid, self.pb.ps_mid, self.pb.vel_mid, self.pb.internalvars_mid
                        else:
                            u, p, v, ivars = self.pb.u, self.pb.p, self.pb.vel, self.pb.internalvars
                        stressfuncs = []
                        for n in range(self.pb.num_domains):
                            stressfuncs.append(self.pb.ma[n].sigma_vonmises(u, p, v, ivar=ivars))
                        vonmises_cauchystress = project(
                            stressfuncs,
                            self.pb.Vd_scalar,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="vonMises_CauchyStress",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        vonmises_cauchystress_out = fem.Function(self.pb.V_out_scalar, name=vonmises_cauchystress.name)
                        vonmises_cauchystress_out.interpolate(vonmises_cauchystress)
                        self.pb.resultsfiles[res].write_function(vonmises_cauchystress_out, indicator)
                    elif res == "pk1stress":
                        if self.pb.io.output_midpoint:
                            u, p, v, ivars = self.pb.us_mid, self.pb.ps_mid, self.pb.vel_mid, self.pb.internalvars_mid
                        else:
                            u, p, v, ivars = self.pb.u, self.pb.p, self.pb.vel, self.pb.internalvars
                        stressfuncs = []
                        for n in range(self.pb.num_domains):
                            stressfuncs.append(self.pb.ma[n].P(u, p, v, ivar=ivars))
                        pk1stress = project(
                            stressfuncs,
                            self.pb.Vd_tensor,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="PK1Stress",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        pk1stress_out = fem.Function(self.pb.V_out_tensor, name=pk1stress.name)
                        pk1stress_out.interpolate(pk1stress)
                        self.pb.resultsfiles[res].write_function(pk1stress_out, indicator)
                    elif res == "pk2stress":
                        if self.pb.io.output_midpoint:
                            u, p, v, ivars = self.pb.us_mid, self.pb.ps_mid, self.pb.vel_mid, self.pb.internalvars_mid
                        else:
                            u, p, v, ivars = self.pb.u, self.pb.p, self.pb.vel, self.pb.internalvars
                        stressfuncs = []
                        for n in range(self.pb.num_domains):
                            stressfuncs.append(self.pb.ma[n].S(u, p, v, ivar=ivars))
                        pk2stress = project(
                            stressfuncs,
                            self.pb.Vd_tensor,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="PK2Stress",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        pk2stress_out = fem.Function(self.pb.V_out_tensor, name=pk2stress.name)
                        pk2stress_out.interpolate(pk2stress)
                        self.pb.resultsfiles[res].write_function(pk2stress_out, indicator)
                    elif res == "jacobian":
                        if self.pb.io.output_midpoint:
                            u = self.pb.us_mid
                        else:
                            u = self.pb.u
                        jacobian = project(
                            self.pb.ki.J(u),
                            self.pb.Vd_scalar,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="Jacobian",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        jacobian_out = fem.Function(self.pb.V_out_scalar, name=jacobian.name)
                        jacobian_out.interpolate(jacobian)
                        self.pb.resultsfiles[res].write_function(jacobian_out, indicator)
                    elif res == "glstrain":
                        if self.pb.io.output_midpoint:
                            u = self.pb.us_mid
                        else:
                            u = self.pb.u
                        glstrain = project(
                            self.pb.ki.E(u),
                            self.pb.Vd_tensor,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="GreenLagrangeStrain",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        glstrain_out = fem.Function(self.pb.V_out_tensor, name=glstrain.name)
                        glstrain_out.interpolate(glstrain)
                        self.pb.resultsfiles[res].write_function(glstrain_out, indicator)
                    elif res == "glstrain_principal":
                        if self.pb.io.output_midpoint:
                            u = self.pb.us_mid
                        else:
                            u = self.pb.u
                        evals, _, _ = spectral_decomposition_3x3(self.pb.ki.E(u))
                        evals_gl = ufl.as_vector(evals)  # written as vector
                        glstrain_principal = project(
                            evals_gl,
                            self.pb.Vd_vector,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="GreenLagrangeStrain_princ",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        glstrain_principal_out = fem.Function(self.pb.V_out_vector, name=glstrain_principal.name)
                        glstrain_principal_out.interpolate(glstrain_principal)
                        self.pb.resultsfiles[res].write_function(glstrain_principal_out, indicator)
                    elif res == "eastrain":
                        if self.pb.io.output_midpoint:
                            u = self.pb.us_mid
                        else:
                            u = self.pb.u
                        eastrain = project(
                            self.pb.ki.e(u),
                            self.pb.Vd_tensor,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="EulerAlmansiStrain",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        eastrain_out = fem.Function(self.pb.V_out_tensor, name=eastrain.name)
                        eastrain_out.interpolate(eastrain)
                        self.pb.resultsfiles[res].write_function(eastrain_out, indicator)
                    elif res == "eastrain_principal":
                        if self.pb.io.output_midpoint:
                            u = self.pb.us_mid
                        else:
                            u = self.pb.u
                        evals, _, _ = spectral_decomposition_3x3(self.pb.ki.e(u))
                        evals_ea = ufl.as_vector(evals)  # written as vector
                        eastrain_principal = project(
                            evals_gl,
                            self.pb.Vd_vector,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="EulerAlmansiStrain_princ",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        eastrain_principal_out = fem.Function(self.pb.V_out_vector, name=eastrain_principal.name)
                        eastrain_principal_out.interpolate(eastrain_principal)
                        self.pb.resultsfiles[res].write_function(eastrain_principal_out, indicator)
                    elif res == "strainenergy":
                        if self.pb.io.output_midpoint:
                            u, p, v, ivars = self.pb.us_mid, self.pb.ps_mid, self.pb.vel_mid, self.pb.internalvars_mid
                        else:
                            u, p, v, ivars = self.pb.u, self.pb.p, self.pb.vel, self.pb.internalvars
                        sefuncs = []
                        for n in range(self.pb.num_domains):
                            sefuncs.append(
                                self.pb.ma[n].S(
                                    u,
                                    p,
                                    v,
                                    ivar=ivars,
                                    returnquantity="strainenergy",
                                )
                            )
                        se = project(
                            sefuncs,
                            self.pb.Vd_scalar,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="StrainEnergy",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        se_out = fem.Function(self.pb.V_out_scalar, name=se.name)
                        se_out.interpolate(se)
                        self.pb.resultsfiles[res].write_function(se_out, indicator)
                    elif res == "internalpower":
                        if self.pb.io.output_midpoint:
                            u, p, v, ivars = self.pb.us_mid, self.pb.ps_mid, self.pb.vel_mid, self.pb.internalvars_mid
                        else:
                            u, p, v, ivars = self.pb.u, self.pb.p, self.pb.vel, self.pb.internalvars
                        pwfuncs = []
                        for n in range(self.pb.num_domains):
                            pwfuncs.append(
                                ufl.inner(
                                    self.pb.ma[n].S(
                                        u,
                                        p,
                                        v,
                                        ivar=ivars,
                                    ),
                                    self.pb.ki.Edot(u, v),
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
                    elif res == "fiberstretch":
                        if self.pb.io.output_midpoint:
                            u = self.pb.us_mid
                        else:
                            u = self.pb.u
                        fiberstretch = project(
                            self.pb.ki.fibstretch(u, self.pb.fib_func[0]),
                            self.pb.Vd_scalar,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="FiberStretch",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        fiberstretch_out = fem.Function(self.pb.V_out_scalar, name=fiberstretch.name)
                        fiberstretch_out.interpolate(fiberstretch)
                        self.pb.resultsfiles[res].write_function(fiberstretch_out, indicator)
                    elif res == "fiberstretch_e":
                        if self.pb.io.output_midpoint:
                            u, theta = self.pb.us_mid, self.pb.internalvars_mid["theta"]
                        else:
                            u, theta = self.pb.u, self.pb.internalvars["theta"]
                        stretchfuncs = []
                        for n in range(self.pb.num_domains):
                            if self.pb.mat_growth[n]:
                                stretchfuncs.append(self.pb.ma[n].fibstretch_e(self.pb.ki.C(u), theta, self.pb.fib_func[0]))
                            else:
                                stretchfuncs.append(ufl.as_ufl(0))
                        fiberstretch_e = project(
                            stretchfuncs,
                            self.pb.Vd_scalar,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="FiberStretch_e",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        fiberstretch_e_out = fem.Function(self.pb.V_out_scalar, name=fiberstretch_e.name)
                        fiberstretch_e_out.interpolate(fiberstretch_e)
                        self.pb.resultsfiles[res].write_function(fiberstretch_e_out, indicator)
                    elif res == "theta":
                        if self.pb.io.output_midpoint:
                            theta_proj = project(
                                self.pb.internalvars_mid["theta"],
                                self.pb.Vd_scalar,
                                self.pb.dx,
                                domids=self.pb.domain_ids,
                                nm="theta",
                                comm=self.pb.comm,
                                entity_maps=self.pb.io.entity_maps,
                            )
                            theta_out = fem.Function(self.pb.V_out_scalar, name=theta_proj.name)
                            theta_out.interpolate(theta_proj)
                        else:
                            theta_out = fem.Function(self.pb.V_out_scalar, name=self.pb.theta.name)
                            theta_out.interpolate(self.pb.theta)
                        self.pb.resultsfiles[res].write_function(theta_out, indicator)
                    elif res == "phi_remod":
                        if self.pb.io.output_midpoint:
                            theta = self.pb.internalvars_mid["theta"]
                        else:
                            theta = self.pb.internalvars["theta"]
                        phifuncs = []
                        for n in range(self.pb.num_domains):
                            if self.pb.mat_remodel[n]:
                                phifuncs.append(self.pb.ma[n].phi_remod(theta))
                            else:
                                phifuncs.append(ufl.as_ufl(0))
                        phiremod = project(
                            phifuncs,
                            self.pb.Vd_scalar,
                            self.pb.dx,
                            domids=self.pb.domain_ids,
                            nm="phiRemodel",
                            comm=self.pb.comm,
                            entity_maps=self.pb.io.entity_maps,
                        )
                        phiremod_out = fem.Function(self.pb.V_out_scalar, name=phiremod.name)
                        phiremod_out.interpolate(phiremod)
                        self.pb.resultsfiles[res].write_function(phiremod_out, indicator)
                    elif res == "tau_a":
                        if self.pb.io.output_midpoint:
                            tau_proj = project(
                                self.pb.internalvars_mid["tau_a"],
                                self.pb.Vd_scalar,
                                self.pb.dx,
                                domids=self.pb.domain_ids,
                                nm="tau_a",
                                comm=self.pb.comm,
                                entity_maps=self.pb.io.entity_maps,
                            )
                            tau_out = fem.Function(self.pb.V_out_scalar, name=tau_proj.name)
                            tau_out.interpolate(tau_proj)
                        else:
                            tau_out = fem.Function(self.pb.V_out_scalar, name=self.pb.tau_a.name)
                            tau_out.interpolate(self.pb.tau_a)
                        self.pb.resultsfiles[res].write_function(tau_out, indicator)
                    elif res == "fibers":
                        # written only once at the beginning, not after each time step (since constant in time)
                        pass
                    else:
                        raise NameError("Unknown output to write for solid mechanics!")

    def readcheckpoint(self, N_rest):
        vecs_to_read = {}
        vecs_to_read[self.pb.u] = "u"
        if self.pb.incompressible_2field:
            vecs_to_read[self.pb.p] = "p"
        if self.pb.is_poroelastic:
            vecs_to_read[self.pb.pporo] = "pporo"
        if any(self.pb.mat_growth) and isinstance(self.pb.theta, fem.function.Function):
            vecs_to_read[self.pb.theta] = "theta"
            vecs_to_read[self.pb.theta_old] = "theta"
        if any(self.pb.mat_active_stress):
            vecs_to_read[self.pb.tau_a] = "tau_a"
            vecs_to_read[self.pb.tau_a_old] = "tau_a"
            if self.pb.have_frank_starling:
                vecs_to_read[self.pb.amp_old] = "amp_old"
        if self.pb.u_pre is not None:
            vecs_to_read[self.pb.u_pre] = "u_pre"

        if self.pb.timint != "static":
            vecs_to_read[self.pb.u_old] = "u"
            vecs_to_read[self.pb.v_old] = "v_old"
            vecs_to_read[self.pb.a_old] = "a_old"
            if self.pb.incompressible_2field:
                vecs_to_read[self.pb.p_old] = "p"
            if self.pb.is_poroelastic:
                vecs_to_read[self.pb.pporo_old] = "pporo"

        if self.pb.pbase.problem_type == "solid_flow0d_multiscale_gandr":
            vecs_to_read[self.pb.u_set] = "u_set"
            vecs_to_read[self.pb.growth_thres] = "growth_thres"
            if self.pb.incompressible_2field:
                vecs_to_read[self.pb.p_set] = "p_set"
            if any(self.pb.mat_active_stress):
                vecs_to_read[self.pb.tau_a_set] = "tau_a_set"
                if self.pb.have_frank_starling:
                    vecs_to_read[self.pb.amp_old_set] = "amp_old_set"

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
                self.pb.io.readfunction(
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
        vecs_to_write[self.pb.u] = "u"
        if self.pb.incompressible_2field:
            vecs_to_write[self.pb.p] = "p"
        if self.pb.is_poroelastic:
            vecs_to_write[self.pb.pporo] = "pporo"
        if any(self.pb.mat_growth) and isinstance(self.pb.theta, fem.function.Function):
            vecs_to_write[self.pb.theta] = "theta"
        if any(self.pb.mat_active_stress):
            vecs_to_write[self.pb.tau_a] = "tau_a"
            if self.pb.have_frank_starling:
                vecs_to_write[self.pb.amp_old] = "amp_old"
        if self.pb.u_pre is not None:
            vecs_to_write[self.pb.u_pre] = "u_pre"

        if self.pb.timint != "static":
            vecs_to_write[self.pb.v_old] = "v_old"
            vecs_to_write[self.pb.a_old] = "a_old"

        if self.pb.pbase.problem_type == "solid_flow0d_multiscale_gandr":
            vecs_to_write[self.pb.u_set] = "u_set"
            vecs_to_write[self.pb.growth_thres] = "growth_thres"
            if self.pb.incompressible_2field:
                vecs_to_write[self.pb.p_set] = "p_set"
            if any(self.pb.mat_active_stress):
                vecs_to_write[self.pb.tau_a_set] = "tau_a_set"
                if self.pb.have_frank_starling:
                    vecs_to_write[self.pb.amp_old_set] = "amp_old_set"

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
                self.pb.io.writefunction(
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
