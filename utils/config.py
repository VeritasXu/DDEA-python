#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

######################################################################
# Non-commercial
# project link: git@github.com:VeritasXu/DDEA-python.git
# official link: https://github.com/HandingWang/DDEA-SE.git
######################################################################

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int,    default=100, help="rounds of training")

    # ensemble arguments
    parser.add_argument('--num_users', type=int, default=300, help="number of users: K")
    parser.add_argument('--num_model', type=int, default=100, help='num of selected model')

    #optimization arguments
    parser.add_argument('--num_pop', type=int,   default=100, help='number of populations')
    parser.add_argument('--boot_prob',type=float,default=0.5, help='bootstrap probability')
    parser.add_argument('--lb', type=float,      default=-5.12, help='lower boundary')
    parser.add_argument('--ub', type=float,      default=5.12, help='upper boundary')
    # Test function arguments
    parser.add_argument('--func_name', type=str, default='Ellipsoid', help='test function name')
    parser.add_argument('--num_b4_d', type=int,  default=11, help='number before dimension, e.g., 11 of 11d')
    parser.add_argument('--d', type=int,         default=10, help='dimension of decision space/number of centers')
    parser.add_argument('--d_out', type=int,     default=1, help="number of output dimension")
    # other arguments
    parser.add_argument('--seed', type=int,      default=1, help='random seed (default: 1)')
    args = parser.parse_args()
    return args
