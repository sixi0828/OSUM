source /mnt/sfs/asr/.bashrc
datetime_str=$(date +"%Y-%-m-%-d_%H")
my_named_nohup train_base-version_only-emotion_$datetime_str bash run_huawei_2p_master_only_emotion.sh
tail -f output/run_res/train_base-version_only-emotion_$datetime_str.nohup
