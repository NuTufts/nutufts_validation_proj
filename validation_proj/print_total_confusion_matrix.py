# print_total_confusion_matrix.py
# 
# The purpose of this script is to take a .npy file produced by
# create_mlreco_validation_analysis_data.py and print the
# total confusion matrix for all voxels of all events contained in it
# for testing
#
# Author: Noah Stiegler
# April 2023

# parse command line arguments
import argparse
parser = argparse.ArgumentParser(description = "Quickly look at total confusion matrix in .npy file output from create_mlreco_validation_analysis_data.py")
parser.add_argument('path', help="Path to the .npy file to look at") # Just has 1 argument
args = parser.parse_args()

# Import numpy
import numpy as np

with open(args.path, 'rb') as f:
    result = np.load(f, allow_pickle=True) # allow_pickle can be a security risk if you
                                           # open a file you didn't create

# Print some info
print("") # Make a new line
print(f"File ({args.path}) had {len(result)} events in it")
print("") # Make a new line
print(f"Data came from root file: {result[0]['file']}")
print("") # Make a new line
print(f"Checkpoint used was: {result[0]['checkpoint']}")
print("") # Make a new line

# Get confusion matrices
non_cosmics = [event["confusion_matrices"]["non_cosmics"] for event in result]
cosmics = [event["confusion_matrices"]["cosmics"] for event in result]

# Make total confusion matrix
all_non_cosmics = sum(non_cosmics)
all_cosmics = sum(cosmics)
total = all_non_cosmics + all_cosmics

print(total)

