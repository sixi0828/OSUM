source /mnt/sfs/asr/.bashrc
datetime_str=$(date +"%Y-%-m-%-d_%H")
my_named_nohup train_base-version_rank1_$datetime_str bash run_huawei_2p_rank1.sh
echo "train_base-version_rank1_$datetime_str.nohup is running"
tail -f output/run_res/train_base-version_rank1_$datetime_str.nohup
