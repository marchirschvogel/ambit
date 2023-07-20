#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def check_params_io(params):

    valid_params = ['duplicate_mesh_domains',
                    'fiber_data',
                    'indicate_results_by',
                    'mesh_domain',
                    'mesh_boundary',
                    'meshfile_type',
                    'order_fib_input',
                    'output_midpoint_0D',
                    'output_path',
                    'output_path_0D',
                    'output_path_pre',
                    'problem_type',
                    'restart_io_type',
                    'restart_step',
                    'results_to_write',
                    'simname',
                    'USE_MIXED_DOLFINX_BRANCH',
                    'use_model_order_red',
                    'write_results_every',
                    'write_results_every_0D',
                    'write_restart_every']

    for k in params.keys():
        if k not in valid_params:
            raise RuntimeError("Unknown parameter found in I/O params: "+k)


def check_params_fem_solid(params):

    valid_params = ['incompressible_2field',
                    'order_disp',
                    'order_pres',
                    'pressure_at_midpoint',
                    'prestress_from_file',
                    'prestress_initial',
                    'prestress_initial_only',
                    'prestress_maxtime',
                    'prestress_numstep',
                    'prestress_ptc',
                    'quad_degree']

    for k in params.keys():
        if k not in valid_params:
            raise RuntimeError("Unknown parameter found in solid FEM params: "+k)


def check_params_fem_fluid(params):

    valid_params = ['initial_fluid_pressure',
                    'fluid_formulation',
                    'order_vel',
                    'order_pres',
                    'pressure_at_midpoint',
                    'prestress_from_file',
                    'prestress_initial',
                    'prestress_initial_only',
                    'prestress_maxtime',
                    'prestress_numstep',
                    'prestress_ptc',
                    'quad_degree',
                    'stabilization']

    for k in params.keys():
        if k not in valid_params:
            raise RuntimeError("Unknown parameter found in fluid FEM params: "+k)


def check_params_fem_ale(params):

    valid_params = ['order_disp',
                    'quad_degree']

    for k in params.keys():
        if k not in valid_params:
            raise RuntimeError("Unknown parameter found in ALE FEM params: "+k)
