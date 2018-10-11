#! /usr/bin/env bash

gfortran -O3 -fPIC -c pygyro/splines/mod_context_1.f90 -o pygyro/splines/mod_context_1.o  -J pygyro/splines/

CC=gcc FC=gfortran  f2py -c  --opt='-O3'  -m  pygyro.splines.spline_eval_funcs pygyro/splines/spline_eval_funcs.f90 pygyro/splines/mod_context_1.o -Ipygyro/splines/

gfortran -O3 -fPIC -c pygyro/initialisation/mod_initialiser_funcs.f90 -o pygyro/initialisation/mod_initialiser_funcs.o  -J pygyro/initialisation/

CC=gcc FC=gfortran  f2py -c  --opt='-O3'  -m  pygyro.initialisation.initialiser_func pygyro/initialisation/initialiser_func.f90 pygyro/initialisation/mod_initialiser_funcs.o -Ipygyro/initialisation/

gfortran -O3 -fPIC -c pygyro/splines/mod_spline_eval_funcs.f90 -o pygyro/splines/mod_spline_eval_funcs.o -J pygyro/splines/

CC=gcc FC=gfortran  f2py -c  --opt='-O3'  -m  pygyro.advection.accelerated_advection_steps pygyro/advection/accelerated_advection_steps.f90 pygyro/initialisation/mod_initialiser_funcs.o -Ipygyro/initialisation/ pygyro/splines/mod_spline_eval_funcs.o -Ipygyro/splines/
