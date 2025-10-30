#!/usr/bin/env python3

# Copyright (c) 2019-2025, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.


def check_params_io(params):
    valid_params = [
        "duplicate_mesh_domains",
        "fiber_data",
        "domain_ids_solid",
        "domain_ids_fluid",
        "gridname_domain",
        "gridname_boundary",
        "indicate_results_by",
        "initial_fields",
        "mesh_dim",
        "mesh_domain",
        "mesh_boundary",
        "mesh_edge",
        "mesh_point",
        "mesh_encoding",
        "mesh_format",
        "ode_parallel",
        "order_fib_input",
        "output_midpoint_0D",
        "output_path",
        "output_path_0D",
        "output_path_pre",
        "print_enhanced_info",
        "problem_type",
        "restart_io_type",
        "restart_step",
        "results_to_write",
        "simname",
        "surface_ids_interface",
        "volume_laplace",
        "write_results_every",
        "write_results_every_0D",
        "write_restart_every",
        "write_submeshes",
    ]

    for k in params.keys():
        if k not in valid_params:
            raise RuntimeError("Unknown parameter found in I/O params: " + k)


def check_params_ctrl(params):
    valid_params = [
        "dt",
        "maxtime",
        "numstep",
        "numstep_stop",
        "prestress_dt",
        "prestress_maxtime",
        "prestress_numstep",
        "residual_scale",
    ]

    for k in params.keys():
        if k not in valid_params:
            raise RuntimeError("Unknown parameter found in control params: " + k)


def check_params_solver(params):
    valid_params = [
        "block_precond",
        "block_precond_prestr",
        "catch_max_res_value",
        "direct_solver",
        "divergence_continue",
        "fieldsplit_type",
        "indexset_options",
        "iterative_solver",
        "k_ptc_initial",
        "lin_norm_type",
        "max_liniter",
        "maxiter",
        "petsc_options_ksp",
        "precond_fields",
        "precond_fields_prestr",
        "print_liniter_every",
        "print_local_iter",
        "ptc",
        "ptc_randadapt_range",
        "res_lin_monitor",
        "rebuild_prec_every_it",
        "schur_block_scaling",
        "solve_type",
        "solve_type_prestr",
        "subsolver_params",
        "tol_inc",
        "tol_inc_local",
        "tol_lin_abs",
        "tol_lin_rel",
        "tol_res",
        "tol_res_local",
    ]

    for k in params.keys():
        if k not in valid_params:
            raise RuntimeError("Unknown parameter found in solver params: " + k)


def check_params_fem_solid(params):
    valid_params = [
        "bulkmod",
        "incompressibility",
        "inverse_mechanics",
        "lin_remodeling_full",
        "order_disp",
        "order_pres",
        "prestress_from_file",
        "prestress_initial",
        "prestress_initial_only",
        "prestress_ptc",
        "quad_degree",
    ]

    for k in params.keys():
        if k not in valid_params:
            raise RuntimeError("Unknown parameter found in solid FEM params: " + k)


def check_params_fem_fluid(params):
    valid_params = [
        "initial_fluid_pressure",
        "fluid_formulation",
        "order_vel",
        "order_pres",
        "prestress_from_file",
        "prestress_initial",
        "prestress_initial_only",
        "prestress_kinetic",
        "prestress_ptc",
        "quad_degree",
        "stabilization",
    ]

    for k in params.keys():
        if k not in valid_params:
            raise RuntimeError("Unknown parameter found in fluid FEM params: " + k)


def check_params_fem_ale(params):
    valid_params = ["order_disp", "quad_degree"]

    for k in params.keys():
        if k not in valid_params:
            raise RuntimeError("Unknown parameter found in ALE FEM params: " + k)


def check_params_fem_cahnhilliard(params):
    valid_params = ["order_phi", "order_mu", "quad_degree"]

    for k in params.keys():
        if k not in valid_params:
            raise RuntimeError("Unknown parameter found in Cahn-Hilliard FEM params: " + k)


def check_params_fem_electrophysiology(params):
    valid_params = ["order_phi", "quad_degree"]

    for k in params.keys():
        if k not in valid_params:
            raise RuntimeError("Unknown parameter found in electrophysiology FEM params: " + k)


def check_params_time_solid(params):
    valid_params = [
        "alpha_m",
        "alpha_f",
        "beta",
        "eval_nonlin_terms",
        "gamma",
        "rho_inf_genalpha",
        "timint",
        "theta_ost",
    ]

    for k in params.keys():
        if k not in valid_params:
            raise RuntimeError("Unknown parameter found in solid time params: " + k)


def check_params_time_fluid(params):
    valid_params = [
        "alpha_m",
        "alpha_f",
        "eval_nonlin_terms",
        "fluid_governing_type",
        "gamma",
        "rho_inf_genalpha",
        "timint",
        "theta_ost",
    ]

    for k in params.keys():
        if k not in valid_params:
            raise RuntimeError("Unknown parameter found in fluid time params: " + k)


def check_params_time_cahnhilliard(params):
    valid_params = [
        "eval_nonlin_terms",
        "timint",
        "theta_ost",
    ]

    for k in params.keys():
        if k not in valid_params:
            raise RuntimeError("Unknown parameter found in Cahn-Hilliard time params: " + k)


def check_params_time_electrophysiology(params):
    valid_params = ["eval_nonlin_terms", "timint", "theta_ost"]

    for k in params.keys():
        if k not in valid_params:
            raise RuntimeError("Unknown parameter found in electrophysiology time params: " + k)


def check_params_time_flow0d(params):
    valid_params = [
        "eps_periodic",
        "initial_backwardeuler",
        "initial_conditions",
        "initial_file",
        "periodic_checktype",
        "timint",
        "theta_ost",
    ]

    for k in params.keys():
        if k not in valid_params:
            raise RuntimeError("Unknown parameter found in flow0d time params: " + k)


def check_params_coupling_fluid_ale(params):
    valid_params = [
        "coupling_ale_fluid",
        "coupling_fluid_ale",
        "coupling_strategy",
        "fsi_governing_type",
        "fsi_system",
        "remove_mutual_solid_fluid_bcs",
    ]

    for k in params.keys():
        if k not in valid_params:
            raise RuntimeError("Unknown parameter found in fluid-ALE coupling params: " + k)


def check_params_rom(params):
    valid_params = [
        "eigenvalue_cutoff",
        "exclude_from_snap",
        "filetype",
        "hdmfilenames",
        "modes_from_files",
        "numredbasisvec",
        "numredbasisvec_partition",
        "numsnapshots",
        "orthogonalize_rom_basis",
        "partitions",
        "pod_only",
        "print_eigenproblem",
        "regularizations",
        "regularizations_integ",
        "regularizations_deriv",
        "snapshotincr",
        "surface_rom",
        "write_pod_modes",
    ]

    for k in params.keys():
        if k not in valid_params:
            raise RuntimeError("Unknown parameter found in ROM params: " + k)
