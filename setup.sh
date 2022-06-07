#!/bin/bash

_parent_dir=$(cd $(dirname ${BASH_SOURCE})/.. && pwd)

source ${_parent_dir}/ibc/venv/bin/activate
export PYTHONPATH=${_parent_dir}
export TF_CPP_MIN_LOG_LEVEL=2
