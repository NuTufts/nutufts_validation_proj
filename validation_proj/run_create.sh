
# $1 is first input, $2 is second input, etc


script=create_mlreco_validation_analysis_data.py
config=/home/nstieg01/lartpc_mlreco3d/config/train_ubmlreco_uresnet_ppn.cfg
snapshot=/home/nstieg01/nWorking/validation_proj/checkpoints/snapshot-619999.ckpt
data=/home/nstieg01/nWorking/validation_proj/validation_data/mlrecodata_bnb_nu_0560.root
outputfilename=scriptoutput.out



python3 $script $config $snapshot $data $outputfilename
