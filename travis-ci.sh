#!/bin/bash
# Copyright © 2017-2018 Martin Ueding <dev@martin-ueding.de>
# Modified by Markus Werner <markus.werner@uni-bonn.de>
# Licensed under the MIT/Expat license.

set -e
set -u
set -x

sourcedir="$(pwd)"
outdir="$HOME/output"

cd ..

###############################################################################
#                              Install Packages                               #
###############################################################################

ubuntu_packages=(
  libhdf5-dev
)

sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install -y "${ubuntu_packages[@]}"

##########################################################################################
# Run tests                                                                              #
##########################################################################################

pushd "$sourcedir"
python integration-test.py
popd
