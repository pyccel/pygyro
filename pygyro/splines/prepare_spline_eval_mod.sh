#!/usr/bin/env bash

pyccel mod_context_1.py -t
gfortran  -fPIC -O3  -c   mod_context_1.f90
python3 spline_eval_mod_creator.py
