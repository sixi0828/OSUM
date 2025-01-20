# shellcheck disable=SC1090
source ~/.bashrc

dir=/home/node54_tmpdata/xlgeng/ckpt/asr_sot_emotion_caption_task_mix/epoch_1
ckpt_name=step_5999.pt
#
#dir=/home/node54_tmpdata/xlgeng/ckpt/asr_sot_emotion_task_mix/epoch_0
#ckpt_name=step_2499.pt

# 推理for sot
test_data_dir=/home/work_nfs15/asr_data/data/asr_test_sets
data_type=shard_full_data # raw  shard_full_data

test_sets="aishell4 ami"
gpu_id=4
task=sot_task # transcribe  sot_task  emotion_task
lang=zh  # en zh
test_sets="${test_sets// /---}"
# shellcheck disable=SC2027
echo "在"$gpu_id"gpu设备上进行"$task"任务的推理,推理数据集为:""$test_sets"
my_named_nohup sot_001 bash decode_common.sh --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task $task --lang $lang

test_sets="alimeeting"
gpu_id=5
task=sot_task # transcribe  sot_task  emotion_task
lang=zh  # en zh
test_sets="${test_sets// /---}"
# shellcheck disable=SC2027
echo "在"$gpu_id"gpu设备上进行"$task"任务的推理,推理数据集为:""$test_sets"
my_named_nohup sot_002 bash decode_common.sh --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task $task --lang $lang
