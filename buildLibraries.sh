#! /usr/bin/env bash

pyccel pygyro/initialisation/mod_initialiser_funcs.py --output pygyro/initialisation --fflags ' -O3 -fPIC'
pyccel pygyro/splines/mod_context_1.py --output pygyro/splines --fflags ' -O3 -fPIC'
python3 moduleGenerator.py
