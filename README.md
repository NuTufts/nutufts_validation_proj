# A collection of LArTPC mlreco3d Inference & Validation Analysis tools & learnings
Author: Noah Stiegler

Created: April 2023

### Overview
This repository contains code used for running inference & validation on the [lartpc\_mlreco3d](https://github.com/NuTufts/lartpc_mlreco3d) neural network trained on simulated MicroBooNE data for the Tufts Neutrino Group (specifically by the Wongjirad Lab). 
The point of the network is to help in the pipeline of turning raw data from a LArTPC detector into information we can analyze to do physics. Specifically, it looks at where energy was deposited in voxels in the detector and tries to figure out which voxels are part of particle events, what particles were involved in the event, and how they may have interacted. 
The network was originally trained on the PILArNet open data set ([Domine et al 2021](https://journals.aps.org/prd/pdf/10.1103/PhysRevD.104.032004)). This dataset contains simulated events which are non-detector specific. 
The goal of the work here for the Wongjirad Lab is to help in the training of the network on data simulated to look like that from the MicroBooNE detector. 
The scripts in this repository use checkpoints (weights and biases) from a network training run, the configuration file it was trained with, and validation data the network has never seen to analyze how it performs on data it's never seen as it trained. These statistics help us understand whether the network works and when it starts to overfit as well as allowing us to quantify how different training techniques and parameters affect the network's performance. 

The ultimate goal of these projects is to better understand neutrinos by collecting data about them. Neutrinos are fundamental particles which interact through the weak force and gravity. They interact rarely, so they're very hard to detect. We don't fully understand neutrinos and how they fit in the standard model, and we hope these experiments will help us learn new physics.

### Files:
- `validation\_proj`:  ipython notebooks and scripts used in running analysis on validation data from the training run of the network
  - `notebooks`: A folder containing all documented interactive notebooks which walk through interacting with the network & data
    - `example\_inference.ipynb`: Reference notebook (not documented)
    - `load\_plot\_analysis.ipynb`: Notebook walking through loading the data saved by the `create\_mlreco...` script and plotting confusion matrices from it
    - `Look At Training Run Statistics.ipynb`: Notebook walking through loading and looking at the saved statistics from the network's training
    - `ubmlreco\_inference.ipynb`: Reference notebook (not documented)
    - `validation\_analysis.ipynb` Notebook walking through loading the lartpc\_mlreco3d network and running inference on it
    - `many\_analyses\_plots.ipynb`: Extension of load\_plot\_analysis notebook for creating plots from the inference and analysis of many checkpoints on many validation files. Takes many .npy saved files and loads them to make plots.
  - `slurm`: A folder containing scripts for running jobs on the tufts cluster using slurm - the job scheduler. 
    - `slurm\_submit\_script.sh`: The script given to the sbatch command for the slurm scheduler. Starts a singularity container and runs the `slurm\_run\_script.sh` script in each of them. Uses an array of values to create several different jobs with different parameters. The array specifies different checkpoints for each sub-job to analyze with all data files in `data\_files.txt`
    - `slurm\_run\_script.sh`: The script run in a singularity container by each job. Actually runs the `create\_...` script. Runs on a single checkpoint for all validation files in `data\_files.txt`
    - `targeted\_slurm\_submit\_script.sh`: Same as the `slurm\_submit` script but modified to only run for a few checkpoints. Useful if a few of the jobs failed and need to be re-run. 
  - `create\_mlreco\_validation\_analysis\_data.py`: The main script of the repository - runs inference over all events in a .root file given a checkpoint and config
  - `run\_create.sh`: A helper script to run the above Python script and pass in arguments
  - `print\_total\_confusion\_matrix.py`: A quick debug script to run on a .npy file created by the `create\_mlreco...` script in this directory. Prints to terminal the total confusion matrix for voxels in all events in an analyzed file.
  - `full\_data\_files.txt`: A running list of all validation data files we've run inference on so far. Used in many\_analyses\_plots.ipynb to know which .npy analyzed files exist.
  - `data\_files.txt`: A list of validation files to run inference on when running the slurm run & submit scripts
- old: some old files from earlier stages of the project of working with lartpc\_mlreco3d (such as voxelizing spacepoint data and training the network) which are no longer needed.

### Dependencies:
- [lartpc\_mlreco3d](https://github.com/NuTufts/lartpc_mlreco3d)
  - MinkowskiEngine
  - torch\_geometric
  - torch\_scatter
  - torch\_sparse
  - torch\_cluster
  - numba
  - scikit-learn
  - networkx
  - pandas
  - possibly more
- [icdl](https://github.com/NuTufts/icdl)
- matplotlib
- possibly more

### Setting up lartpc\_mlreco3d
Follow these steps:
1) `git clone https://github.com/NuTufts/lartpc\_mlreco3d`
2) Run inside a singularity container which has MinkowskiEngine setup (or in an environment already set up)
3) May need to install some new python packages if it throws a PackageNotFoundError. use `pip install --user [package-name]` to install it locally 

### Using lartpc\_mlreco3d
The lartpc\_mlreco3d network can be worked with almost entirely in python. To run, it needs a config (.cfg) file. For an example of running inference, see /validation\_proj/notebooks/validation\_analysis.ipynb

### Parts of lartpc\_mlreco3d (that we've trained so far)
- UResNet: One of the first parts of the network, used in initial voxel classification
- PPN: Point Proposal Network - tries to figure out track and shower start and end points

### Setting up icdl
Follow these steps:
1) `git clone https://github.com/NuTufts/icdl.git`
1a) `cd icdl` (make sure you're in the icdl directory)
2) `git submodule init` (because some of the folders in the repository are submodules which point to folders in other repositories, those folders are empty)
3) `git submodule update` (fill empty folders)
4) `source setenv\_py3.sh` (in icdl folder, sets up environment variables)
5) `source configure.sh` (more shell configuration)
6) `source buildall\_py3.sh` (builds all the dependencies such as ROOT, OpenCV, Python. Takes a while)

**Note**: steps 4 & 5 will need to be repeated each time you log on / use icdl environment in a fresh shell

### Environments on the Tufts Cluster
For running lartpc\_mlreco3d
1) Log into Tufts HPC
2) Start an interactive compute node with a GPU.
2a) `srun --pty -p wongjiradlab --mem-per-cpu=4000 --cpus-per-task=2 --gres=gpu --time 2:00:00 bash`
3) Load a singularity container with some of the packages needed.
3a) `module load singularity/3.5.3`
3b) `singularity run --nv --bind /cluster/tufts/:/cluster/tufts/,/tmp:/tmp /cluster/tufts/wongjiradlabnu/larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_comput8.sif)`
4) Locally install any other needed Python packages - see Setting Up lartpc\_mlreco3d section above
5) `alias python=python3` to make sure we're using python3 not python2
6) `source /cluster/home/nstieg01/scripts/setup_icdl.sh` to set up the icdl environment with ROOT
6a) Alternatively run `source setenv_py3.sh` and `source configure.sh` in the icdl directory. The above command just runs these files
7) Run the network!

For using jupyter notebook to plot (such as with `many\_analyses\_plots.ipynb`):
1) Log into Tufts HPC
2) Start an interactive compute node and write down the name of the node. It should be the new prompt. For example: p1cmp075
2a) `srun --pty -p wongjiradlab --mem-per-cpu=4000 --cpus-per-task=2 --gres=gpu --time 2:00:00 bash`
3) Load a singularity environment with jupyter notebooks in it. I've been using `/cluster/tufts/minos/jwolcott/data/singularity-images/ub20.04-cuda11.0-pytorch1.7.1-larndsim.sif`
3a) The script `/cluster/home/nstieg01/scripts/setup_singularity_old.sh` will do exactly this
4) Make sure you're in the Bash shell by typing `bash` if you see the singularity prompt
5) Setup the icdl environment. See step 6 above for running lartpc\_mlreco3d: `source /cluster/home/nstieg01/scripts/setup_icdl.sh`
6) Start a jupyter notebook on a specific port on the compute node: `jupyter notebook --no-browser --port=8003`
7) Open a different terminal window on the computer you want to use jupyter notebook
8) Forward the port from the compute node to your computer with a command like `ssh -tt username@login.cluster.tufts.edu -L 8003:localhost:8003 ssh NODE -L 8003:localhost:8003`, where NODE is replaced with the name of the compute node from step 2
9) Go back to the first terminal window and copy a link which jupyter notebook should put in the terminal in step 6 into a browser on the computer you want to work on, you should now have access! 


### Data
The data we're using lartpc\_mlreco3d with is LArTPC data in the ROOT & ICDL format. 
ROOT & ICDL are software packages, and LArTPC stands for Liquid Argon Time Projection Chamber. This type of detector detects energy deposited by particles as they move through the detector and ionize the liquid argon inside it. Because the detector is in an electric field, the freed electrons from ionization float to one side of the chamber, where they're recorded by a grid of wires. 
So data is recorded as three 2D wireplane images in the MicroBooNE detector. The Neutrino Group has a method of reconstructing the 3d event from the wireplane images, which puts them into a sparse spacepoint format (in ROOT), which then need to be converted into a sparse voxel format (in ROOT) to use with lartpc\_mlreco3d. This reconstruction process isn't perfect, and creates ghost points (which are reconstructed points which don't actually exist, kind of like noise). In the ICDL simulated data, a particle label of 5 means a ghost point. 
Because the data is simulated, we know almost everything about it, so the data also contains information about every voxel such as what particle it came from, what "cluster" it belongs to (group of voxels) and whether it came from cosmic background or a real neutrino event.
The config files we use tell mlreco the format of the data in the ROOT file we give it. The ROOT files which contain the data often have 50-100 events. 
The main create\_mlreco\_validation\_analysis_data.py script in this repository runs inference on all events in a ROOT file. Hypothetically, the network can take many ROOT files as input. 

#### Classification Labels
##### From UResNet
- 0 = protons
- 1 = MIPs (minimum ionizing particles like muon or pion)
- 2 = EM showers (electron, positron, photon)
- 3 = Delta ray electrons (hard scattering off of charged particles)
- 4 = Michel electrons (decay of muons)  
- (5 = ghost)
##### From later in the network chain
- 0 = photon
- 1 = e+/-
- 2 = Mu +/-
- 3 = pi+/-
- 4 = protons


### Related Tutorials
- https://github.com/NuTufts/icdl/wiki
- https://github.com/NuTufts/lartpc\_mlreco3d
- http://deeplearnphysics.org/Blog/tutorial\_summary.html#tutorial\_summary
- https://deeplearnphysics.org/lartpc\_mlreco3d\_tutorials/


