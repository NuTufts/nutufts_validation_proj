#!/bin/bash

module load singularity/3.5.3

singulary shell -nv -B /cluster:/cluster /cluster/tufts/wongjiradlabnu/larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_compute8_wjupyternotebook.sif
