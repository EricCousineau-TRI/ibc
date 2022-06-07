#!/bin/bash

_parent_dir=$(cd $(dirname ${BASH_SOURCE})/.. && pwd)

source ${_parent_dir}/ibc/venv/bin/activate
export PYTHONPATH=${_parent_dir}
export CUDA_VISIBLE_DEVICES=1
export TF_CPP_MIN_LOG_LEVEL=2
