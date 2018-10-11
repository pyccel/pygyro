#! /usr/bin/env bash

ifort -O3 -fPIC -c pygyro/splines/mod_context_1.f90 -o pygyro/splines/mod_context_1.o  -module pygyro/splines/

CC=icc  f2py -c  --opt='-O3'  -m  pygyro.splines.spline_eval_funcs pygyro/splines/spline_eval_funcs.f90 pygyro/splines/mod_context_1.o -Ipygyro/splines/ --fcompiler=intelem

ifort -O3 -fPIC -c pygyro/initialisation/mod_initialiser_funcs.f90 -o pygyro/initialisation/mod_initialiser_funcs.o  -module pygyro/initialisation/

CC=icc  f2py -c  --opt='-O3'  -m  pygyro.initialisation.initialiser_func pygyro/initialisation/initialiser_func.f90 pygyro/initialisation/mod_initialiser_funcs.o -Ipygyro/initialisation/ --fcompiler=intelem

ifort -O3 -fPIC -c pygyro/splines/mod_spline_eval_funcs.f90 -o pygyro/splines/mod_spline_eval_funcs.o -module pygyro/splines/

CC=icc  f2py -c  --opt='-O3'  -m  pygyro.advection.accelerated_advection_steps pygyro/advection/accelerated_advection_steps.f90 pygyro/initialisation/mod_initialiser_funcs.o -Ipygyro/initialisation/ pygyro/splines/mod_spline_eval_funcs.o -Ipygyro/splines/ --fcompiler=intelem
