gxl_data_json_info_path=conf/data_config_huawei.yaml

data_path=/mnt/sfs/asr/data
data=$data_path/qwen2_multi_task_4_6gpus_gxl_adapter
mkdir -p $data

data_type=shard_full_data
train_data=data/tmp/tmp.list
python gxl_utils/handle_data_for_weight.py $gxl_data_json_info_path $train_data