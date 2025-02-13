source /mnt/sfs/asr/.bashrc
datetime_str=$(date +"%Y-%-m-%-d_%H")
my_named_nohup train_base-version_rank3_$datetime_str bash run_huawei_2p_rank3.sh
echo "train_base-version_rank3_$datetime_str.nohup is running"
tail -f output/run_res/train_base-version_rank3_$datetime_str.nohup
