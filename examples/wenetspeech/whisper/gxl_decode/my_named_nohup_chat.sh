# shellcheck disable=SC1090
source ~/.bashrc
export CUDA_HOME='/usr/local/cuda-11.7'
dir=/home/node54_tmpdata/xlgeng/ckpt/emotion_task_mix/epoch_0
ckpt_name=step_9999.pt   # 需要带pt

dir=/home/node54_tmpdata/xlgeng/ckpt/asr_sot_emotion_caption_task_mix/epoch_1
ckpt_name=step_5999.pt

dir=/home/node54_tmpdata/xlgeng/ckpt/llmasr_stage1/epoch_0
ckpt_name=step_129999.pt

# 推理for asr
test_data_dir=/home/node54_tmpdata/xlgeng/chat_data
data_type=shard_full_data # raw  shard_full_data

test_sets="shards_test"
gpu_id=6
task=transcribe # transcribe  sot_task  emotion_task chat_task
lang=zh  # en zh
test_sets="${test_sets// /---}"
# shellcheck disable=SC2027
echo "在"$gpu_id"gpu设备上进行"$task"任务的推理,推理数据集为:""$test_sets"
#my_named_nohup llmasr_asr_001 bash decode_common.sh --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task $task --lang $lang
bash decode_common.sh --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task $task --lang $lang

#test_sets="test_net_2"
#gpu_id=0
#task=transcribe # transcribe  sot_task  emotion_task
#lang=zh  # en zh
#test_sets="${test_sets// /---}"
## shellcheck disable=SC2027
#echo "在"$gpu_id"gpu设备上进行"$task"任务的推理,推理数据集为:""$test_sets"
#my_named_nohup llmasr_asr_001 bash decode_common.sh --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task $task --lang $lang
#
