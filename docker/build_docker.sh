#!/bin/bash

echo ""
echo "Building image m1tutorial_parameter_tuning"

if [[ ! $1 ]]; then
    echo "You need to specify as password for this image, like so: sudo docker_build.sh my_password"
    echo "exiting build script"
    exit 1
fi
if [[ ! $USER ]]; then
    echo "You USER environment variable is not set"
    echo "exiting build script"
    exit 1
fi
if [[ ! $UID  ]]; then
    echo "You UID environment variable is not set"
    echo "exiting build script"
    exit 1
fi

docker build -t m1tutorial_parameter_tuning \
       --build-arg uname=$USER --build-arg uid=$(id -u) --build-arg gid=$(id -g) --build-arg password=$1 \
       .