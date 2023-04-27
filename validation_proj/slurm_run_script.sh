#!/bin/bash

# Print SLURM diagnostic
JOBSTARTDATE=$(date)
echo "running job number ${SLURM_JOB_ID} with array ID ${SLURM_ARRAY_TASK_ID} on node ${SLURMD_NODENAME}"

# Define what this script is doing
# Which snapshot are we on? Array goes from 0-9
# Snapshots go 9999-759999, increasing by 10,000
# Want to go from 129,999 to 759,999 increasing by 70,000
FILE_LIST="/cluster/home/nstieg01/nutufts_validation_proj/validation_proj/data_files.txt"
PATH_TO_SNAPSHOT="/cluster/tufts/wongjiradlabnu/twongj01/mlreco/lartpc_mlreco3d/arxiv/train_with_segmentweights/"
let snapshot_num="${SLURM_ARRAY_TASK_ID} * 70000 + 129999"
SNAPSHOT="${PATH_TO_SNAPSHOT}snapshot-{snapshot_num}.ckpt"
echo "Using snapshot: ${SNAPSHOT}"

# Loop from Matthew Rosenberg @ Tufts
PATH_TO_DATA="/cluster/tufts/wongjiradlabnu/twongj01/mlreco/icdl/larflow/larmatchnet/dataprep/mlreco_data/valid/"
maxFileCount=`wc -l < $FILE_LIST`
let firstfile="1"
let lastfile="$maxFileCount"

for n in $(seq $firstfile $lastfile); do
  if (($n > $maxFileCount)); then
    break
  fi

  newfile=`sed -n ${n}p ${FILE_LIST}`

  echo "Processing: ${PATH_TO_DATA}${newfile}"
done


# Print more slurm diagnostic
JOBENDDATE=$(date)
echo "Job began at $JOBSTARTDATE"
echo "Job ended at $JOBENDDATE"
