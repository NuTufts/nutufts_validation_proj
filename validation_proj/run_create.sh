# Trex
# config=/home/nstieg01/lartpc_mlreco3d/config/train_ubmlreco_uresnet_ppn.cfg
# snapshot=/home/nstieg01/nWorking/validation_proj/checkpoints/snapshot-619999.ckpt
# data=/home/nstieg01/nWorking/validation_proj/validation_data/mlrecodata_bnb_nu_0560.root
# outputfilename=scriptoutput.out

# Cluster
script=create_mlreco_validation_analysis_data.py
config=/cluster/tufts/wongjiradlabnu/twongj01/mlreco/lartpc_mlreco3d/config/train_ubmlreco_uresnet_ppn.cfg
snapshot=/cluster/tufts/wongjiradlabnu/twongj01/mlreco/lartpc_mlreco3d/arxiv/train_with_segmentweights/snapshot-619999.ckpt
data=/cluster/tufts/wongjiradlabnu/twongj01/mlreco/icdl/larflow/larmatchnet/dataprep/mlreco_data/valid/mlrecodata_bnb_nu_0560.root
outputfilename=/cluster/home/nstieg01/validation_proj/analysis1out.npy

# Run
python3 $script $config $snapshot $data $outputfilename
