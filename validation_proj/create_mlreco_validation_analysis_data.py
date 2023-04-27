# create_mlreco_validation_analysis_data.py
#
# Last Modified 4/17/23
# Author: Noah Stiegler
#
# Purpose: Determine how lartpc_mlreco3d performs at given checkpoints on 
#          given validation files, saving data for later analysis
#
# Usage: python3 create_mlreco_validation_analysis_data.py path_to_cfg
#                                                          path_to_checkpoint
#                                                          path_to_data
#                                                          name_of_file_to_save_data_in
#                                                          [-v --verbose]


# THIS LINE MAY NEED TO BE CHANGED BASED ON THE SYSTEM THE SCRIPT IS RUN ON
# it should point to the local lartpc_mlreco3d clone
################################################
# SOFTWARE_DIR = "/home/nstieg01/lartpc_mlreco3d" # trex <------ Change this if necessary
SOFTWARE_DIR = "/cluster/home/nstieg01/lartpc_mlreco3d" # Cluster
# SOFTWARE_DIR = "/cluster/tufts/wongjiradlabnu/twongj01/mlreco/lartpc_mlreco3d" # Other cluster from debugging
###############################################

# For reference
#----------------------------------------------
#            Snapshot Locations
# /cluster/tufts/wongjiradlabnu/twongj01/mlreco/lartpc_mlreco3d/arxiv/train_with_segmentweights # from cluster
# /home/nstieg01/nWorking/validation_proj/checkpoints # trex
# snapshots at range(9999, 760000, 10000)
# format: snapshot-___.ckpt
#----------------------------------------------

#----------------------------------------------
#           Validation Data Locations
# /cluster/tufts/wongjiradlabnu/twongj01/mlreco/icdl/larflow/larmatchnet/dataprep/mlreco_data/valid # on cluster
# /home/nstieg01/nWorking/validation_proj/validation_data # trex
# indices: 540-572
# format: mlrecodata_bnb_nu_0___.root
# indices: 460-492
# format: mlrecodata_bnbnue_0___.root
#----------------------------------------------

# Check if lartpc_mlreco3d exists at SOFTWARE_DIR
import os
assert(os.path.isdir(SOFTWARE_DIR)), f"lartpc_mlreco3d not found at {SOFTWARE_DIR} (hardcoded in this script). Change this path if necessary."

# Tell system where mlreco is
import sys
sys.path.insert(0, SOFTWARE_DIR)

# Parse command line arguments
import argparse

# Initialize parser
msg = ''' create_mlreco_validation_analysis_data.py is a script to determine how lartpc_mlreco3d performs at given checkpoints on given validation files, saving data for later analysis. Script may need to be edited'''
parser = argparse.ArgumentParser(description = msg)

# Add positional arguments
parser.add_argument('cfg_path', help="Path to the cfg file used to load the network")
parser.add_argument('checkpoint_path', help="Path to the checkpoint saved from the network")
parser.add_argument('data_path', help="Path to the validation data to analyze the network with")
parser.add_argument('output_filename', help="Name of the file to save the output of the script in")

# Add flags
parser.add_argument('-v', '--verbose', action='store_true')

# Parse arguments
args = parser.parse_args()

# Check if files exist
assert(os.path.isfile(args.cfg_path)), f"Checkpoint not found at {args.cfg_path}"
assert(os.path.isfile(args.checkpoint_path)), f"Checkpoint not found at {args.checkpoint_path}"
assert(os.path.isfile(args.data_path)), f"Data not found at {args.data_path}"

# Import everything else we'll need
import numpy as np
import yaml
from larcv import larcv

# The cluster doesn't have sklearn :(
try:
    from sklearn.metrics import confusion_matrix
except:
    # If it doesn't have it, define our own
    # Takes in the true labels and the predicted labels
    # If labels is set, looks for those labels in data
    # If not, expects true and predicted to be integers,
    #     where some label is 0, 1, 2, ... n - 1 labels
    def confusion_matrix(true, predicted, labels=None):
        # Figure out size of confusion matrix to make
        if labels != None:
            K = len(labels)
            # If there are labels, create a map from label->index in matrix 
            labelmap = {}
            for i, label in enumerate(labels):
                labelmap[label] = i
        else:
            K = len(np.unique(true)) # Number of classes 
        
        # Create empty confusion matrix
        result = np.zeros((K, K), dtype=np.int64)

        # Fill out confusion matrix
        if labels == None:
            for i in range(len(true)):
                result[true[i]][predicted[i]] += 1
        else:
            for i in range(len(true)):
                result[labelmap[true[i]]][labelmap[predicted[i]]] += 1
        
        # Return confusion matrix
        return result


# Import mlreco
if args.verbose:
    print("Importing mlreco")
from mlreco.main_funcs import process_config, prepare

# Prepare Network
# Load the config from a yaml file format into cfg which is a dictionary
if args.verbose:
    print("Loading network config file")
with open(args.cfg_path) as file:
    config = file.read()
    cfg=yaml.load(config, Loader=yaml.Loader)

# Implement changes we need
# Stop from training
cfg["trainval"]["train"] = False

# Raise voxel cutoff to avoid issues
cfg["iotool"]["dataset"]["nvoxel_limit"] = 1000000

# Set what the weights and biases are
cfg['trainval']['model_path'] = args.checkpoint_path 

cfg['iotool']['dataset']['data_keys'] = [args.data_path]

# Make batchsize 1
cfg['iotool']['batchsize'] = 1

# Change to a sequential sampler to get all data with no repeats
cfg["iotool"]["sampler"]["name"] = "SequentialBatchSampler"

# Add a new parser to look at cosmic_origin
cfg["iotool"]["dataset"]["schema"]["cosmic_origin"] = {"parser": "parse_sparse3d", "args": {"sparse_event_list": ["sparse3d_cosmic_origin"]}}

# If on the cluster, this can be [0]
# If on trex, has to be []
# IF ON TREX:  Set to run on no GPUs because otherwise it crashes -- hopefully we find a fix soon
# Note: works on cluster
cfg["trainval"]["gpus"] = [0] # 

# Make sure the cfg has what we need
schema = cfg["iotool"]["dataset"]["schema"]
assert("input_data" in schema), "Error in config. iotool->dataset->schema->input_data does not exist."
assert("segment_label" in schema), "Error in config. iotool->dataset->schema->segment_label does not exist."

# pre-process configuration (checks + certain non-specified default settings)
# prepare function configures necessary "handlers"
# handlers are what we use to do things in the network
if args.verbose:
    process_config(cfg, verbose=True)
    hs=prepare(cfg)
else:
    # Stop from printing a bunch of stuff
    old_stdout = sys.stdout # backup current stdout
    sys.stdout = open(os.devnull, "w") # change stdout

    # Run functions which print
    process_config(cfg, verbose=False)
    hs=prepare(cfg)

    # Allow printing again
    sys.stdout = old_stdout # reset old stdout

# Num of entries to run on
num_entries = len(hs.data_io) # run on all of them, all of the entries it has
batch = 0 # Batchsize = 1, so for some reason it returns everything inside an array of length 1

# Setup numpy array to save
to_save = []

# Loop over all entries in file
# for entry in range(0, 3): # just do 3 entries for testing
for entry in range(0, num_entries):
    # Inform which entry
    print(".............................")
    print("Running on entry:", entry)


    # If we're verbose, let hs.trainer.forward spew info, if not, stop it from doing so
    if args.verbose:
        data, output = hs.trainer.forward(hs.data_io_iter)
    else:
        # Stop from printing a bunch of stuff
        old_stdout = sys.stdout # backup current stdout
        sys.stdout = open(os.devnull, "w") # change stdout

        # Run functions which print
        data, output = hs.trainer.forward(hs.data_io_iter)

        # Allow printing again
        sys.stdout = old_stdout # reset old stdout

    print("Size: ", len(data["input_data"][batch])) 
    print(".............................")

    # Setup to save everything from this event
    result = {}
    result["index"] = data["index"]
    result["file"] = args.data_path
    result["checkpoint"] = args.checkpoint_path

    # Save easy info
    for key in output:
        contains = output[key][batch]
        key_type = type(contains)
        if (key_type == type(0.1) or key_type == type(1)):
            result[key] = contains

    # Make confusion matrices
    input_data = data["input_data"][batch]
    true_id_labels = data["segment_label"][batch][:, -1]
    not_cosmic = data['cosmic_origin'][batch][:, 4] == 0
    cosmic = data['cosmic_origin'][batch][:, 4] == 1
    predicted_ghost_full = output['ghost'][batch]
    predicted_ghost = predicted_ghost_full.argmax(axis=1)
    predicted_id_labels_full = output['segmentation'][batch]
    predicted_id_labels = predicted_id_labels_full.argmax(axis=1)
    true_ghost_mask = true_id_labels == 5
    predicted_ghost_mask = predicted_ghost == 1
    full_predicted_labels = np.array([predicted_id_labels[i] if (predicted_ghost[i] != 1) else 5 for i in range(0, len(predicted_id_labels))])

    # Set up the dictionary to put in result
    confusion_matrices = {}
    bins = [0, 1, 2, 3, 4, 5]
    confusion_matrices["bins"] = bins
    confusion_matrices["labels"] = ["proton", "MIPs", "e$^-$/e$^+$/$\gamma$", "$\Delta$-ray", "Michel", "ghost"]

    # Sometimes the lengths don't match -- indicates we cut off the number of voxels
    # If it's a cutoff event, just ignore it
    if ((len(true_id_labels) != len(cosmic)) or (len(full_predicted_labels) != len(cosmic))):
        print("lens didn't match")
    else:
        # Cosmics    
        cm = confusion_matrix(true_id_labels[cosmic], full_predicted_labels[cosmic], labels=bins)
        confusion_matrices["cosmics"] = cm

        # Non-cosmics
        cm = confusion_matrix(true_id_labels[not_cosmic], full_predicted_labels[not_cosmic], labels=bins)
        confusion_matrices["non_cosmics"] = cm

        result["confusion_matrices"] = confusion_matrices

        # Save result
        to_save.append(result)
    
# Save to_save to a file
np.save(args.output_filename, to_save)

print(f"Done - file saved to {args.output_filename}")
