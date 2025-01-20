# shellcheck disable=SC1090
# source ~/.bashrc
# export CUDA_HOME='/usr/local/cuda-11.7'
. ./path.sh
export LD_LIBRARY_PATH=/usr/lib/:$LD_LIBRARY_PATH

dir=/home/node54_tmpdata/yzli/wenet_whisper_finetune/examples/wenetspeech/whisper/exp/qwen2_asr_6gpus
ckpt_name=epoch_9.pt

# dir=/home/node54_tmpdata/xlgeng/ckpt/llmasr_stage1/epoch_2
# ckpt_name=step_144999.pt

# 推理for asr
test_data_dir=/home/work_nfs15/asr_data/data/asr_test_sets
data_type=raw # raw  shard_full_data

test_sets="test_meeting"
gpu_id=0
task=transcribe # transcribe  sot_task  emotion_task
lang=zh  # en zh
test_sets="${test_sets// /---}"
# shellcheck disable=SC2027
echo "在"$gpu_id"gpu设备上进行"$task"任务的推理,推理数据集为:""$test_sets"
#my_named_nohup llmasr_asr_001 bash decode_common.sh --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task $task --lang $lang
my_named_nohup llmasr_asr_000$ckpt_name bash decode_common.sh --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task $task --lang $lang

test_sets="test_net_2"
gpu_id=1
task=transcribe # transcribe  sot_task  emotion_task
lang=zh  # en zh
test_sets="${test_sets// /---}"
# shellcheck disable=SC2027
echo "在"$gpu_id"gpu设备上进行"$task"任务的推理,推理数据集为:""$test_sets"
my_named_nohup llmasr_asr_001$ckpt_name bash decode_common.sh --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task $task --lang $lang

test_sets="test_net_1"
gpu_id=2
task=transcribe # transcribe  sot_task  emotion_task chat_task
lang=zh  # en zh
test_sets="${test_sets// /---}"
# shellcheck disable=SC2027
echo "在"$gpu_id"gpu设备上进行"$task"任务的推理,推理数据集为:""$test_sets"
my_named_nohup llmasr_asr_002$ckpt_name bash decode_common.sh --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task $task --lang $lang

test_sets="aishell2"
gpu_id=3
task=transcribe # transcribe  sot_task  emotion_task
lang=zh  # en zh
test_sets="${test_sets// /---}"
# shellcheck disable=SC2027
echo "在"$gpu_id"gpu设备上进行"$task"任务的推理,推理数据集为:""$test_sets"
my_named_nohup llmasr_asr_003$ckpt_name bash decode_common.sh --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task $task --lang $lang

test_sets="aishell1 speechio_0 speechio_4"
gpu_id=4
task=transcribe # transcribe  sot_task  emotion_task
lang=zh  # en zh
test_sets="${test_sets// /---}"
# shellcheck disable=SC2027
echo "在"$gpu_id"gpu设备上进行"$task"任务的推理,推理数据集为:""$test_sets"
my_named_nohup llmasr_asr_004$ckpt_name bash decode_common.sh --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task $task --lang $lang

test_sets="speechio_1 speechio_2 speechio_3"
gpu_id=6
task=transcribe # transcribe  sot_task  emotion_task
lang=zh  # en zh
test_sets="${test_sets// /---}"
# shellcheck disable=SC2027
echo "在"$gpu_id"gpu设备上进行"$task"任务的推理,推理数据集为:""$test_sets"
my_named_nohup llmasr_asr_005$ckpt_name bash decode_common.sh --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task $task --lang $lang


test_sets="speechio_4 speechio_5 speechio_6"
gpu_id=5
task=transcribe # transcribe  sot_task  emotion_task
lang=zh  # en zh
test_sets="${test_sets// /---}"
# shellcheck disable=SC2027
echo "在"$gpu_id"gpu设备上进行"$task"任务的推理,推理数据集为:""$test_sets"
my_named_nohup llmasr_asr_006$ckpt_name bash decode_common.sh --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task $task --lang $lang

#
#test_sets="aishell1"
#gpu_id=7
#task=transcribe # transcribe  sot_task  emotion_task
#lang=zh  # en zh
#test_sets="${test_sets// /---}"
## shellcheck disable=SC2027
#echo "在"$gpu_id"gpu设备上进行"$task"任务的推理,推理数据集为:""$test_sets"
#my_named_nohup asr0055 bash decode_common.sh --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task $task --lang $lang
