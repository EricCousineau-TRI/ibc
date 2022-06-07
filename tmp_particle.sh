#!/bin/bash
cd $(dirname ${BASH_SOURCE})/..
source ./ibc/setup.sh

set -eu
./ibc/ibc/configs/particle/collect_data.sh
./ibc/ibc/configs/particle/run_mlp_ebm_langevin.sh 2
