#!/bin/bash
# Copyright © 2017-2018 Martin Ueding <dev@martin-ueding.de>
# Modified by Markus Werner <markus.werner@uni-bonn.de>
# Licensed under the MIT/Expat license.

set -e
set -u
set -x

sourcedir="$(pwd)"
outdir="$(HOME)/output"

cd ..

###############################################################################
#                              Install Packages                               #
###############################################################################

ubuntu_packages=(
    python
    python-scipy 
    python-matplotlib 
    python-pandas 
    python-sympy 
)

sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install -y "${ubuntu_packages[@]}"

##########################################################################################
# Setup output folder                                                                    #
##########################################################################################

mkdir -p "$(outdir)/integration/3_gevp-data"
cp "$(sourcedir)/tests/integration/*.ini" "$(outdir)/integration/3_gevp-data/"

##########################################################################################
# Run tests                                                                              #
##########################################################################################

pushd $(sourcedir)
python analyse -i tests/integration/A40.24.ini -vv
popd
