# shellcheck disable=SC1090
source ~/.bashrc

dir=/home/node54_tmpdata/xlgeng/ckpt/emotion_task_mix/epoch_0
ckpt_name=step_9999.pt   # 需要带pt

dir=/home/node54_tmpdata/xlgeng/ckpt/asr_sot_task_mix/epoch_3
ckpt_name=step_17999.pt

# 推理for emotion
data_type=shard_full_data # raw  shard_full_data
test_sets="casia cremad esd iemocap_4 meld"
test_data_dir=/home/node54_tmpdata/xlgeng/emotion_data/shards_test/shards_test
gpu_id=0
task=emotion_task # transcribe  sot_task  emotion_task
lang=zh  # en zh
test_sets="${test_sets// /---}"
# shellcheck disable=SC2027
#echo "在"$gpu_id"gpu设备上进行"$task"任务的推理,推理数据集为:""$test_sets"
#bash decode_common.sh --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task $task --lang $lang

# 推理for asr
test_data_dir=/home/work_nfs15/asr_data/data/asr_test_sets
data_type=raw # raw  shard_full_data

test_sets="test_net_1 aishell1"
gpu_id=0
task=transcribe # transcribe  sot_task  emotion_task
lang=zh  # en zh
test_sets="${test_sets// /---}"
# shellcheck disable=SC2027
echo "在"$gpu_id"gpu设备上进行"$task"任务的推理,推理数据集为:""$test_sets"
my_named_nohup asr001 bash decode_common.sh --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task $task --lang $lang

test_sets="test_net_2 aishell2"
gpu_id=1
task=transcribe # transcribe  sot_task  emotion_task
lang=zh  # en zh
test_sets="${test_sets// /---}"
# shellcheck disable=SC2027
echo "在"$gpu_id"gpu设备上进行"$task"任务的推理,推理数据集为:""$test_sets"
my_named_nohup asr002 bash decode_common.sh --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task $task --lang $lang

test_sets="test_meeting"
gpu_id=2
task=transcribe # transcribe  sot_task  emotion_task
lang=zh  # en zh
test_sets="${test_sets// /---}"
# shellcheck disable=SC2027
echo "在"$gpu_id"gpu设备上进行"$task"任务的推理,推理数据集为:""$test_sets"
my_named_nohup asr003 bash decode_common.sh --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task $task --lang $lang

