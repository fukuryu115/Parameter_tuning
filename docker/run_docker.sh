#!/bin/bash

PARENT_DIR=$(pwd|xargs dirname)

docker run --name m1tutorial_parameter_tuning -it --rm -d --gpus all --ipc=host \
           --ulimit memlock=-1 --ulimit stack=67108864 \
           -v /tmp/.X11-unix:/tmp/.X11-unix: -v "${PARENT_DIR}":/home/"${USER}"/Parameter_tuning \
           -e DISPLAY="${DISPLAY}" \
           -p 62222:62222 \
           -p 6001:6001 \
           -p 6002:6002 \
           m1tutorial_parameter_tuning jupyter lab --config='./Parameter_tuning/jupyter_lab_config.py'
