
#!/bin/bash

scriptLocation=/home/nstieg01/icdl/larflow/larmatchnet/dataprep/make_mlrecodata_from_larcv.py
fileToRun=/tutorial_files/dlmerged_larflowtruth_mcc9_v13_bnbnue_corsika_run00001_subrun00001.root
wireOverlapLocation=/tutorial_files/output_microboone_wireoverlap_matrices.root
detector=uboone
outputname=1_event_test_mlrecodata_from_larcv_output.root

python $scriptLocation -d $detector --input-larlite $fileToRun --input-larcv $fileToRun -o $outputname -adc wire -n 1 -e 0 -wo $wireOverlapLocation
