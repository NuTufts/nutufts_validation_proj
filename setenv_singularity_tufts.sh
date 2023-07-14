#!/bin/bash

# Get the folder we called this script from. We will try to return here at the end of the script
NU_VALID_PROJ_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Set the configuration folder for jupyter.
# The default one is defined in the readonly part of the container
export JUPYTER_CONFIG_DIR=~/jupyter/
mkdir -p $JUPYTER_CONFIG_DIR

ICDL_FOLDER=""
ICDL_FOLDER="/cluster/tufts/wongjiradlabnu/twongj01/mlreco/icdl/"

[[ -z "$ICDL_FOLDER" ]] && { echo "You need to set the icdl folder to the ICDL_FOLDER variable in this script."; return; }

cd $ICDL_FOLDER
source setenv_py3.sh
source configure.sh

cd $NU_VALID_PROJ_HOME

echo "Setup singularity container shell environment"
echo "defined JUPYTER_CONFIG_DIR=${JUPYTER_CONFIG_DIR}"
echo "Ran icdl setenv_py3.sh and configure.sh"
echo "The above setup: "
echo "root: $(root-config --bindir)"
echo "larlite: ${LARLITE_BASEDIR}"
echo "larcv: ${LARCV_BASEDIR}"




