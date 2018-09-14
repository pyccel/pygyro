#! /usr/bin/env bash
rm pygyro/splines/*.so
python3 pygyro/advection/accelerated_advection_steps.py
python3 pygyro/splines/spline_eval_funcs.py
