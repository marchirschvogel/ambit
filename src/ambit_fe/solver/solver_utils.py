#!/usr/bin/env python3

# Copyright (c) 2019-2026, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from .. import utilities


class sol_utils:
    def __init__(self, solver):
        self.solver = solver

    def catch_solver_errors(self, resnorm=0, incnorm=0, maxresval=1e16, maxincval=1e16, linconv=1, report=True):
        err = 0

        if np.isnan(resnorm):
            if report: utilities.print_status("NaN encountered.", self.solver.comm)
            err = 1

        if resnorm >= maxresval:
            if report: utilities.print_status("Large residual > max val %.1E encountered." % (maxresval), self.solver.comm)

            err = 1

        if incnorm >= maxincval:
            if report: utilities.print_status("Large increment > max val %.1E encountered." % (maxincval), self.solver.comm)
            err = 1

        if np.isinf(incnorm):
            if report: utilities.print_status("Inf encountered.",self.solver.comm)
            err = 1

        if linconv < 0:  # values smaller 0 indicate divergence of PETSc ksp method
            if report: utilities.print_status("Linear solver diverged.", self.solver.comm)
            err = 1

        return err

    def print_nonlinear_iter(
        self,
        it=0,
        resnorms=None,
        incnorms=None,
        header=False,
        ts=0,
        te=0,
        sub=False,
        ptype=None,
        pi=0,
    ):

        if ptype == "flow0d" or ptype == "signet":
            eqs, vrs = self.solver.pb.eq_names, self.solver.pb.var_names
        else:
            eqs, vrs = self.solver.pb[pi].eq_names, self.solver.pb[pi].var_names
        numres = len(eqs)

        if sub:
            assert(numres==1)

        formatstringhead1 = "{:<" + str(self.solver.indlen) + "s}{:<6s}{:<25s}{:<3s}{:<7s}"
        formatstringhead2 = "{:<" + str(self.solver.indlen) + "s}{:<6s}{:<13s}{:<12s}{:<3s}{:<10s}{:<7s}"
        for j in range(1,numres):
            formatstringhead1 = formatstringhead1.replace("{:<3s}{:<7s}", "{:<3s}{:<25s}")
            formatstringhead1 += "{:<3s}{:<7s}"
            formatstringhead2 = formatstringhead2.replace("{:<10s}{:<7s}", "{:<13s}{:<12s}{:<3s}")
            formatstringhead2 += "{:<10s}{:<7s}"

        formatstringvar0 = "{:<" + str(self.solver.indlen) + "s}{:<3d}{:<3s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.2e}{:<2s}{:<8s}"
        for j in range(1,numres):
            formatstringvar0 = formatstringvar0.replace("{:<4.2e}{:<2s}{:<8s}", "{:<4.4e}{:<3s}{:<10s}{:<5s}")
            formatstringvar0 += "{:<4.2e}{:<2s}{:<8s}"

        formatstringvari = "{:<" + str(self.solver.indlen) + "s}{:<3d}{:<3s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.2e}{:<2s}{:<4.2e}"
        for j in range(1,numres):
            formatstringvari = formatstringvari.replace("{:<4.2e}{:<2s}{:<4.2e}", "{:<4.4e}{:<3s}{:<4.4e}{:<5s}")
            formatstringvari += "{:<4.2e}{:<2s}{:<4.2e}"

        # using greek symbol print (Δ) is not supported everywhere... so use d instead
        if header:
            if numres == 1:
                if not sub:
                    utilities.print_status(
                        (formatstringhead1).format(
                            " ", "it |", eqs[0], "| ", "timings"
                        ),
                        self.solver.comm,
                    )
                    utilities.print_status(
                        (formatstringhead2).format(
                            " ",
                            "#  |",
                            "||r_" + vrs[0] + "||_2",
                            "||d" + vrs[0] + "||_2",
                            "| ",
                            "te",
                            "ts",
                        ),
                        self.solver.comm,
                    )
                else:
                    formatstringhead1 = "{:<" + str(self.solver.indlen) + "s}{:<6s}{:<6s}{:<25s}{:<3s}{:<7s}"
                    formatstringhead2 = "{:<" + str(self.solver.indlen) + "s}{:<6s}{:<6s}{:<13s}{:<12s}{:<3s}{:<10s}{:<7s}"

                    utilities.print_status(" ", self.solver.comm)
                    utilities.print_status(
                        "       ****************** 0D model solve ******************",
                        self.solver.comm,
                    )
                    utilities.print_status(
                        (formatstringhead1).format(
                            " ", " ", "it |", eqs[0], "| ", "timings"
                        ),
                        self.solver.comm,
                    )
                    utilities.print_status(
                        (formatstringhead2).format(
                            " ",
                            " ",
                            "#  |",
                            "||r_" + vrs[0] + "||_2",
                            "||d" + vrs[0] + "||_2",
                            "| ",
                            "te",
                            "ts",
                        ),
                        self.solver.comm,
                    )
            elif numres == 2:
                utilities.print_status(
                    (formatstringhead1).format(
                        " ", "it |", eqs[0], "| ", eqs[1], "| ", "timings"
                    ),
                    self.solver.comm,
                )
                utilities.print_status(
                    (formatstringhead2).format(
                        " ",
                        "#  |",
                        "||r_" + vrs[0] + "||_2",
                        "||d" + vrs[0] + "||_2",
                        "| ",
                        "||r_" + vrs[1] + "||_2",
                        "||d" + vrs[1] + "||_2",
                        "| ",
                        "te",
                        "ts",
                    ),
                    self.solver.comm,
                )
            elif numres == 3:
                utilities.print_status(
                    (formatstringhead1).format(
                        " ", "it |", eqs[0], "| ", eqs[1], "| ", eqs[2], "| ", "timings"
                    ),
                    self.solver.comm,
                )
                utilities.print_status(
                    (formatstringhead2).format(
                        " ",
                        "#  |",
                        "||r_" + vrs[0] + "||_2",
                        "||d" + vrs[0] + "||_2",
                        "| ",
                        "||r_" + vrs[1] + "||_2",
                        "||d" + vrs[1] + "||_2",
                        "| ",
                        "||r_" + vrs[2] + "||_2",
                        "||d" + vrs[2] + "||_2",
                        "| ",
                        "te",
                        "ts",
                    ),
                    self.solver.comm,
                )
            elif numres == 4:
                utilities.print_status(
                    (formatstringhead1).format(
                        " ",
                        "it |",
                        eqs[0],
                        "| ",
                        eqs[1],
                        "| ",
                        eqs[2],
                        "| ",
                        eqs[3],
                        "| ",
                        "timings",
                    ),
                    self.solver.comm,
                )
                utilities.print_status(
                    (formatstringhead2).format(
                        " ",
                        "#  |",
                        "||r_" + vrs[0] + "||_2",
                        "||d" + vrs[0] + "||_2",
                        "| ",
                        "||r_" + vrs[1] + "||_2",
                        "||d" + vrs[1] + "||_2",
                        "| ",
                        "||r_" + vrs[2] + "||_2",
                        "||d" + vrs[2] + "||_2",
                        "| ",
                        "||r_" + vrs[3] + "||_2",
                        "||d" + vrs[3] + "||_2",
                        "| ",
                        "te",
                        "ts",
                    ),
                    self.solver.comm,
                )
            elif numres == 5:
                utilities.print_status(
                    (formatstringhead1).format(
                        " ",
                        "it |",
                        eqs[0],
                        "| ",
                        eqs[1],
                        "| ",
                        eqs[2],
                        "| ",
                        eqs[3],
                        "| ",
                        eqs[4],
                        "| ",
                        "timings",
                    ),
                    self.solver.comm,
                )
                utilities.print_status(
                    (formatstringhead2).format(
                        " ",
                        "#  |",
                        "||r_" + vrs[0] + "||_2",
                        "||d" + vrs[0] + "||_2",
                        "| ",
                        "||r_" + vrs[1] + "||_2",
                        "||d" + vrs[1] + "||_2",
                        "| ",
                        "||r_" + vrs[2] + "||_2",
                        "||d" + vrs[2] + "||_2",
                        "| ",
                        "||r_" + vrs[3] + "||_2",
                        "||d" + vrs[3] + "||_2",
                        "| ",
                        "||r_" + vrs[4] + "||_2",
                        "||d" + vrs[4] + "||_2",
                        "| ",
                        "te",
                        "ts",
                    ),
                    self.solver.comm,
                )
            elif numres == 6:
                utilities.print_status(
                    (formatstringhead1).format(
                        " ",
                        "it |",
                        eqs[0],
                        "| ",
                        eqs[1],
                        "| ",
                        eqs[2],
                        "| ",
                        eqs[3],
                        "| ",
                        eqs[4],
                        "| ",
                        eqs[5],
                        "| ",
                        "timings",
                    ),
                    self.solver.comm,
                )
                utilities.print_status(
                    (formatstringhead2).format(
                        " ",
                        "#  |",
                        "||r_" + vrs[0] + "||_2",
                        "||d" + vrs[0] + "||_2",
                        "| ",
                        "||r_" + vrs[1] + "||_2",
                        "||d" + vrs[1] + "||_2",
                        "| ",
                        "||r_" + vrs[2] + "||_2",
                        "||d" + vrs[2] + "||_2",
                        "| ",
                        "||r_" + vrs[3] + "||_2",
                        "||d" + vrs[3] + "||_2",
                        "| ",
                        "||r_" + vrs[4] + "||_2",
                        "||d" + vrs[4] + "||_2",
                        "| ",
                        "||r_" + vrs[5] + "||_2",
                        "||d" + vrs[5] + "||_2",
                        "| ",
                        "te",
                        "ts",
                    ),
                    self.solver.comm,
                )
            elif numres == 7:
                utilities.print_status(
                    (formatstringhead1).format(
                        " ",
                        "it |",
                        eqs[0],
                        "| ",
                        eqs[1],
                        "| ",
                        eqs[2],
                        "| ",
                        eqs[3],
                        "| ",
                        eqs[4],
                        "| ",
                        eqs[5],
                        "| ",
                        eqs[6],
                        "| ",
                        "timings",
                    ),
                    self.solver.comm,
                )
                utilities.print_status(
                    (formatstringhead2).format(
                        " ",
                        "#  |",
                        "||r_" + vrs[0] + "||_2",
                        "||d" + vrs[0] + "||_2",
                        "| ",
                        "||r_" + vrs[1] + "||_2",
                        "||d" + vrs[1] + "||_2",
                        "| ",
                        "||r_" + vrs[2] + "||_2",
                        "||d" + vrs[2] + "||_2",
                        "| ",
                        "||r_" + vrs[3] + "||_2",
                        "||d" + vrs[3] + "||_2",
                        "| ",
                        "||r_" + vrs[4] + "||_2",
                        "||d" + vrs[4] + "||_2",
                        "| ",
                        "||r_" + vrs[5] + "||_2",
                        "||d" + vrs[5] + "||_2",
                        "| ",
                        "||r_" + vrs[6] + "||_2",
                        "||d" + vrs[6] + "||_2",
                        "| ",
                        "te",
                        "ts",
                    ),
                    self.solver.comm,
                )
            elif numres == 8:
                utilities.print_status(
                    (formatstringhead1).format(
                        " ",
                        "it |",
                        eqs[0],
                        "| ",
                        eqs[1],
                        "| ",
                        eqs[2],
                        "| ",
                        eqs[3],
                        "| ",
                        eqs[4],
                        "| ",
                        eqs[5],
                        "| ",
                        eqs[6],
                        "| ",
                        eqs[7],
                        "| ",
                        "timings",
                    ),
                    self.solver.comm,
                )
                utilities.print_status(
                    (formatstringhead2).format(
                        " ",
                        "#  |",
                        "||r_" + vrs[0] + "||_2",
                        "||d" + vrs[0] + "||_2",
                        "| ",
                        "||r_" + vrs[1] + "||_2",
                        "||d" + vrs[1] + "||_2",
                        "| ",
                        "||r_" + vrs[2] + "||_2",
                        "||d" + vrs[2] + "||_2",
                        "| ",
                        "||r_" + vrs[3] + "||_2",
                        "||d" + vrs[3] + "||_2",
                        "| ",
                        "||r_" + vrs[4] + "||_2",
                        "||d" + vrs[4] + "||_2",
                        "| ",
                        "||r_" + vrs[5] + "||_2",
                        "||d" + vrs[5] + "||_2",
                        "| ",
                        "||r_" + vrs[6] + "||_2",
                        "||d" + vrs[6] + "||_2",
                        "| ",
                        "||r_" + vrs[7] + "||_2",
                        "||d" + vrs[7] + "||_2",
                        "| ",
                        "te",
                        "ts",
                    ),
                    self.solver.comm,
                )
            elif numres == 9:
                utilities.print_status(
                    (formatstringhead1).format(
                        " ",
                        "it |",
                        eqs[0],
                        "| ",
                        eqs[1],
                        "| ",
                        eqs[2],
                        "| ",
                        eqs[3],
                        "| ",
                        eqs[4],
                        "| ",
                        eqs[5],
                        "| ",
                        eqs[6],
                        "| ",
                        eqs[7],
                        "| ",
                        eqs[8],
                        "| ",
                        "timings",
                    ),
                    self.solver.comm,
                )
                utilities.print_status(
                    (formatstringhead2).format(
                        " ",
                        "#  |",
                        "||r_" + vrs[0] + "||_2",
                        "||d" + vrs[0] + "||_2",
                        "| ",
                        "||r_" + vrs[1] + "||_2",
                        "||d" + vrs[1] + "||_2",
                        "| ",
                        "||r_" + vrs[2] + "||_2",
                        "||d" + vrs[2] + "||_2",
                        "| ",
                        "||r_" + vrs[3] + "||_2",
                        "||d" + vrs[3] + "||_2",
                        "| ",
                        "||r_" + vrs[4] + "||_2",
                        "||d" + vrs[4] + "||_2",
                        "| ",
                        "||r_" + vrs[5] + "||_2",
                        "||d" + vrs[5] + "||_2",
                        "| ",
                        "||r_" + vrs[6] + "||_2",
                        "||d" + vrs[6] + "||_2",
                        "| ",
                        "||r_" + vrs[7] + "||_2",
                        "||d" + vrs[7] + "||_2",
                        "| ",
                        "||r_" + vrs[8] + "||_2",
                        "||d" + vrs[8] + "||_2",
                        "| ",
                        "te",
                        "ts",
                    ),
                    self.solver.comm,
                )
            elif numres == 10:
                utilities.print_status(
                    (formatstringhead1).format(
                        " ",
                        "it |",
                        eqs[0],
                        "| ",
                        eqs[1],
                        "| ",
                        eqs[2],
                        "| ",
                        eqs[3],
                        "| ",
                        eqs[4],
                        "| ",
                        eqs[5],
                        "| ",
                        eqs[6],
                        "| ",
                        eqs[7],
                        "| ",
                        eqs[8],
                        "| ",
                        eqs[9],
                        "| ",
                        "timings",
                    ),
                    self.solver.comm,
                )
                utilities.print_status(
                    (formatstringhead2).format(
                        " ",
                        "#  |",
                        "||r_" + vrs[0] + "||_2",
                        "||d" + vrs[0] + "||_2",
                        "| ",
                        "||r_" + vrs[1] + "||_2",
                        "||d" + vrs[1] + "||_2",
                        "| ",
                        "||r_" + vrs[2] + "||_2",
                        "||d" + vrs[2] + "||_2",
                        "| ",
                        "||r_" + vrs[3] + "||_2",
                        "||d" + vrs[3] + "||_2",
                        "| ",
                        "||r_" + vrs[4] + "||_2",
                        "||d" + vrs[4] + "||_2",
                        "| ",
                        "||r_" + vrs[5] + "||_2",
                        "||d" + vrs[5] + "||_2",
                        "| ",
                        "||r_" + vrs[6] + "||_2",
                        "||d" + vrs[6] + "||_2",
                        "| ",
                        "||r_" + vrs[7] + "||_2",
                        "||d" + vrs[7] + "||_2",
                        "| ",
                        "||r_" + vrs[8] + "||_2",
                        "||d" + vrs[8] + "||_2",
                        "| ",
                        "||r_" + vrs[9] + "||_2",
                        "||d" + vrs[9] + "||_2",
                        "| ",
                        "te",
                        "ts",
                    ),
                    self.solver.comm,
                )
            else:
                raise RuntimeError("Error. You should not be here!")

            return

        if it == 0:
            if numres == 1:
                if not sub:
                    utilities.print_status(
                        (formatstringvar0).format(
                            " ",
                            it,
                            "| ",
                            resnorms["res1"],
                            " ",
                            " ",
                            "  |  ",
                            te,
                            " ",
                            " ",
                        ),
                        self.solver.comm,
                    )
                else:
                    formatstringvar0 = "{:<" + str(self.solver.indlen) + "s}{:<6s}{:<3d}{:<3s}{:<4.4e}{:<3s}{:<10s}{:<5s}{:<4.2e}{:<2s}{:<8s}"
                    utilities.print_status(
                        (formatstringvar0).format(
                            " ",
                            " ",
                            it,
                            "| ",
                            resnorms["res1"],
                            " ",
                            " ",
                            "  |  ",
                            te,
                            " ",
                            " ",
                        ),
                        self.solver.comm,
                    )
            elif numres == 2:
                utilities.print_status(
                    (formatstringvar0).format(
                        " ",
                        it,
                        "| ",
                        resnorms["res1"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res2"],
                        " ",
                        " ",
                        "  |  ",
                        te,
                        " ",
                        " ",
                    ),
                    self.solver.comm,
                )
            elif numres == 3:
                utilities.print_status(
                    (formatstringvar0).format(
                        " ",
                        it,
                        "| ",
                        resnorms["res1"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res2"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res3"],
                        " ",
                        " ",
                        "  |  ",
                        te,
                        " ",
                        " ",
                    ),
                    self.solver.comm,
                )
            elif numres == 4:
                utilities.print_status(
                    (formatstringvar0).format(
                        " ",
                        it,
                        "| ",
                        resnorms["res1"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res2"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res3"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res4"],
                        " ",
                        " ",
                        "  |  ",
                        te,
                        " ",
                        " ",
                    ),
                    self.solver.comm,
                )
            elif numres == 5:
                utilities.print_status(
                    (formatstringvar0).format(
                        " ",
                        it,
                        "| ",
                        resnorms["res1"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res2"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res3"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res4"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res5"],
                        " ",
                        " ",
                        "  |  ",
                        te,
                        " ",
                        " ",
                    ),
                    self.solver.comm,
                )
            elif numres == 6:
                utilities.print_status(
                    (formatstringvar0).format(
                        " ",
                        it,
                        "| ",
                        resnorms["res1"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res2"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res3"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res4"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res5"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res6"],
                        " ",
                        " ",
                        "  |  ",
                        te,
                        " ",
                        " ",
                    ),
                    self.solver.comm,
                )
            elif numres == 7:
                utilities.print_status(
                    (formatstringvar0).format(
                        " ",
                        it,
                        "| ",
                        resnorms["res1"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res2"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res3"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res4"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res5"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res6"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res7"],
                        " ",
                        " ",
                        "  |  ",
                        te,
                        " ",
                        " ",
                    ),
                    self.solver.comm,
                )
            elif numres == 8:
                utilities.print_status(
                    (formatstringvar0).format(
                        " ",
                        it,
                        "| ",
                        resnorms["res1"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res2"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res3"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res4"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res5"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res6"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res7"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res8"],
                        " ",
                        " ",
                        "  |  ",
                        te,
                        " ",
                        " ",
                    ),
                    self.solver.comm,
                )
            elif numres == 9:
                utilities.print_status(
                    (formatstringvar0).format(
                        " ",
                        it,
                        "| ",
                        resnorms["res1"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res2"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res3"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res4"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res5"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res6"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res7"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res8"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res9"],
                        " ",
                        " ",
                        "  |  ",
                        te,
                        " ",
                        " ",
                    ),
                    self.solver.comm,
                )
            elif numres == 10:
                utilities.print_status(
                    (formatstringvar0).format(
                        " ",
                        it,
                        "| ",
                        resnorms["res1"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res2"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res3"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res4"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res5"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res6"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res7"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res8"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res9"],
                        " ",
                        " ",
                        "  |  ",
                        resnorms["res10"],
                        " ",
                        " ",
                        "  |  ",
                        te,
                        " ",
                        " ",
                    ),
                    self.solver.comm,
                )
            else:
                raise RuntimeError("Number of residual norms inconsistent.")

        else:
            if numres == 1:
                if not sub:
                    utilities.print_status(
                        (formatstringvari).format(
                            " ",
                            it,
                            "| ",
                            resnorms["res1"],
                            " ",
                            incnorms["inc1"],
                            "  |  ",
                            te,
                            " ",
                            ts,
                        ),
                        self.solver.comm,
                    )
                else:
                    formatstringvari = "{:<" + str(self.solver.indlen) + "s}{:<6s}{:<3d}{:<3s}{:<4.4e}{:<3s}{:<4.4e}{:<5s}{:<4.2e}{:<2s}{:<4.2e}"
                    utilities.print_status(
                        (formatstringvari).format(
                            " ",
                            " ",
                            it,
                            "| ",
                            resnorms["res1"],
                            " ",
                            incnorms["inc1"],
                            "  |  ",
                            te,
                            " ",
                            ts,
                        ),
                        self.solver.comm,
                    )
            elif numres == 2:
                utilities.print_status(
                    (formatstringvari).format(
                        " ",
                        it,
                        "| ",
                        resnorms["res1"],
                        " ",
                        incnorms["inc1"],
                        "  |  ",
                        resnorms["res2"],
                        " ",
                        incnorms["inc2"],
                        "  |  ",
                        te,
                        " ",
                        ts,
                    ),
                    self.solver.comm,
                )
            elif numres == 3:
                utilities.print_status(
                    (formatstringvari).format(
                        " ",
                        it,
                        "| ",
                        resnorms["res1"],
                        " ",
                        incnorms["inc1"],
                        "  |  ",
                        resnorms["res2"],
                        " ",
                        incnorms["inc2"],
                        "  |  ",
                        resnorms["res3"],
                        " ",
                        incnorms["inc3"],
                        "  |  ",
                        te,
                        " ",
                        ts,
                    ),
                    self.solver.comm,
                )
            elif numres == 4:
                utilities.print_status(
                    (formatstringvari).format(
                        " ",
                        it,
                        "| ",
                        resnorms["res1"],
                        " ",
                        incnorms["inc1"],
                        "  |  ",
                        resnorms["res2"],
                        " ",
                        incnorms["inc2"],
                        "  |  ",
                        resnorms["res3"],
                        " ",
                        incnorms["inc3"],
                        "  |  ",
                        resnorms["res4"],
                        " ",
                        incnorms["inc4"],
                        "  |  ",
                        te,
                        " ",
                        ts,
                    ),
                    self.solver.comm,
                )
            elif numres == 5:
                utilities.print_status(
                    (formatstringvari).format(
                        " ",
                        it,
                        "| ",
                        resnorms["res1"],
                        " ",
                        incnorms["inc1"],
                        "  |  ",
                        resnorms["res2"],
                        " ",
                        incnorms["inc2"],
                        "  |  ",
                        resnorms["res3"],
                        " ",
                        incnorms["inc3"],
                        "  |  ",
                        resnorms["res4"],
                        " ",
                        incnorms["inc4"],
                        "  |  ",
                        resnorms["res5"],
                        " ",
                        incnorms["inc5"],
                        "  |  ",
                        te,
                        " ",
                        ts,
                    ),
                    self.solver.comm,
                )
            elif numres == 6:
                utilities.print_status(
                    (formatstringvari).format(
                        " ",
                        it,
                        "| ",
                        resnorms["res1"],
                        " ",
                        incnorms["inc1"],
                        "  |  ",
                        resnorms["res2"],
                        " ",
                        incnorms["inc2"],
                        "  |  ",
                        resnorms["res3"],
                        " ",
                        incnorms["inc3"],
                        "  |  ",
                        resnorms["res4"],
                        " ",
                        incnorms["inc4"],
                        "  |  ",
                        resnorms["res5"],
                        " ",
                        incnorms["inc5"],
                        "  |  ",
                        resnorms["res6"],
                        " ",
                        incnorms["inc6"],
                        "  |  ",
                        te,
                        " ",
                        ts,
                    ),
                    self.solver.comm,
                )
            elif numres == 7:
                utilities.print_status(
                    (formatstringvari).format(
                        " ",
                        it,
                        "| ",
                        resnorms["res1"],
                        " ",
                        incnorms["inc1"],
                        "  |  ",
                        resnorms["res2"],
                        " ",
                        incnorms["inc2"],
                        "  |  ",
                        resnorms["res3"],
                        " ",
                        incnorms["inc3"],
                        "  |  ",
                        resnorms["res4"],
                        " ",
                        incnorms["inc4"],
                        "  |  ",
                        resnorms["res5"],
                        " ",
                        incnorms["inc5"],
                        "  |  ",
                        resnorms["res6"],
                        " ",
                        incnorms["inc6"],
                        "  |  ",
                        resnorms["res7"],
                        " ",
                        incnorms["inc7"],
                        "  |  ",
                        te,
                        " ",
                        ts,
                    ),
                    self.solver.comm,
                )
            elif numres == 8:
                utilities.print_status(
                    (formatstringvari).format(
                        " ",
                        it,
                        "| ",
                        resnorms["res1"],
                        " ",
                        incnorms["inc1"],
                        "  |  ",
                        resnorms["res2"],
                        " ",
                        incnorms["inc2"],
                        "  |  ",
                        resnorms["res3"],
                        " ",
                        incnorms["inc3"],
                        "  |  ",
                        resnorms["res4"],
                        " ",
                        incnorms["inc4"],
                        "  |  ",
                        resnorms["res5"],
                        " ",
                        incnorms["inc5"],
                        "  |  ",
                        resnorms["res6"],
                        " ",
                        incnorms["inc6"],
                        "  |  ",
                        resnorms["res7"],
                        " ",
                        incnorms["inc7"],
                        "  |  ",
                        resnorms["res8"],
                        " ",
                        incnorms["inc8"],
                        "  |  ",
                        te,
                        " ",
                        ts,
                    ),
                    self.solver.comm,
                )
            elif numres == 9:
                utilities.print_status(
                    (formatstringvari).format(
                        " ",
                        it,
                        "| ",
                        resnorms["res1"],
                        " ",
                        incnorms["inc1"],
                        "  |  ",
                        resnorms["res2"],
                        " ",
                        incnorms["inc2"],
                        "  |  ",
                        resnorms["res3"],
                        " ",
                        incnorms["inc3"],
                        "  |  ",
                        resnorms["res4"],
                        " ",
                        incnorms["inc4"],
                        "  |  ",
                        resnorms["res5"],
                        " ",
                        incnorms["inc5"],
                        "  |  ",
                        resnorms["res6"],
                        " ",
                        incnorms["inc6"],
                        "  |  ",
                        resnorms["res7"],
                        " ",
                        incnorms["inc7"],
                        "  |  ",
                        resnorms["res8"],
                        " ",
                        incnorms["inc8"],
                        "  |  ",
                        resnorms["res9"],
                        " ",
                        incnorms["inc9"],
                        "  |  ",
                        te,
                        " ",
                        ts,
                    ),
                    self.solver.comm,
                )
            elif numres == 10:
                utilities.print_status(
                    (formatstringvari).format(
                        " ",
                        it,
                        "| ",
                        resnorms["res1"],
                        " ",
                        incnorms["inc1"],
                        "  |  ",
                        resnorms["res2"],
                        " ",
                        incnorms["inc2"],
                        "  |  ",
                        resnorms["res3"],
                        " ",
                        incnorms["inc3"],
                        "  |  ",
                        resnorms["res4"],
                        " ",
                        incnorms["inc4"],
                        "  |  ",
                        resnorms["res5"],
                        " ",
                        incnorms["inc5"],
                        "  |  ",
                        resnorms["res6"],
                        " ",
                        incnorms["inc6"],
                        "  |  ",
                        resnorms["res7"],
                        " ",
                        incnorms["inc7"],
                        "  |  ",
                        resnorms["res8"],
                        " ",
                        incnorms["inc8"],
                        "  |  ",
                        resnorms["res9"],
                        " ",
                        incnorms["inc9"],
                        "  |  ",
                        resnorms["res10"],
                        " ",
                        incnorms["inc10"],
                        "  |  ",
                        te,
                        " ",
                        ts,
                    ),
                    self.solver.comm,
                )
            else:
                raise RuntimeError("Number of residual norms inconsistent.")

    def print_linear_iter(self, it, rnorm):
        if it == 0:
            self.rnorm_start = rnorm
            utilities.print_status(" ", self.solver.comm)
            utilities.print_status(
                ("{:<" + str(self.solver.indlen) + "s}{:<8s}{:<47s}").format(
                    " ", " ", "***************** linear solve ****************"
                ),
                self.solver.comm,
            )

        if it % self.solver.print_liniter_every == 0:
            if self.solver.res_lin_monitor == "rel":
                resnorm = rnorm / self.rnorm_start
            elif self.solver.res_lin_monitor == "abs":
                resnorm = rnorm
            else:
                raise ValueError("Unknown res_lin_monitor value. Choose 'rel' or 'abs'.")

            utilities.print_status(
                ("{:<" + str(self.solver.indlen) + "s}{:<17s}{:<4d}{:<21s}{:<4e}").format(
                    " ",
                    "        lin. it.: ",
                    it,
                    "     " + self.solver.res_lin_monitor + ". res. norm:",
                    resnorm,
                ),
                self.solver.comm,
            )

    def print_linear_iter_last(self, it, rnorm, conv_reason):
        if self.solver.res_lin_monitor == "rel":
            resnorm = rnorm / self.rnorm_start
        elif self.solver.res_lin_monitor == "abs":
            resnorm = rnorm
        else:
            raise ValueError("Unknown res_lin_monitor value. Choose 'rel' or 'abs'.")

        if it % self.solver.print_liniter_every != 0:  # otherwise already printed
            utilities.print_status(
                ("{:<" + str(self.solver.indlen) + "s}{:<17s}{:<4d}{:<21s}{:<4e}").format(
                    " ",
                    "        lin. it.: ",
                    it,
                    "     " + self.solver.res_lin_monitor + ". res. norm:",
                    resnorm,
                ),
                self.solver.comm,
            )
        # cf. https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.KSP.ConvergedReason-class.html for converge codes
        utilities.print_status(
            ("{:<" + str(self.solver.indlen) + "s}{:<8s}{:<13s}{:<18s}{:<2d}{:<14s}").format(
                " ",
                " ",
                "************ ",
                " PETSc conv code: ",
                conv_reason,
                " *************",
            ),
            self.solver.comm,
        )
        if not self.solver.pb[0].print_subiter:
            utilities.print_status(
                " ", self.solver.comm
            )  # no extra line if we have an "intermediate" print from another model... TODO: Find a nicer solution here...

        # update counters
        self.solver.li += it
        self.solver.li_s.append(it)

    def check_converged(self, resnorms, incnorms, tolerances, ptype=None):
        if ptype is None:
            ptype = self.solver.ptype

        converged = []

        len_r = len(resnorms)
        len_i = len(incnorms)
        assert len_r == len_i

        for i in range(len_r):
            if (
                resnorms["res" + str(i + 1)] <= tolerances["res" + str(i + 1)]
                and incnorms["inc" + str(i + 1)] <= tolerances["inc" + str(i + 1)]
            ):
                converged.append(True)
            else:
                converged.append(False)

        return all(converged)

    def timestep_separator_len(self):
        if len(self.solver.tolerances[0]) == 2:
            seplen = 53
        elif len(self.solver.tolerances[0]) == 4:
            seplen = 81
        elif len(self.solver.tolerances[0]) == 6:
            seplen = 109
        elif len(self.solver.tolerances[0]) == 8:
            seplen = 137
        elif len(self.solver.tolerances[0]) == 10:
            seplen = 165
        elif len(self.solver.tolerances[0]) == 12:
            seplen = 195
        elif len(self.solver.tolerances[0]) == 14:
            seplen = 221
        elif len(self.solver.tolerances[0]) == 16:
            seplen = 240
        elif len(self.solver.tolerances[0]) == 18:
            seplen = 260
        elif len(self.solver.tolerances[0]) == 20:
            seplen = 280
        else:
            raise ValueError("Unknown size of tolerances!")

        return seplen
