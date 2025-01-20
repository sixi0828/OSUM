# shellcheck disable=SC1090
source ~/.bashrc

dir=/home/node54_tmpdata/xlgeng/ckpt/emotion_task_mix/epoch_0
ckpt_name=step_9999.pt   # 需要带pt
dir=/home/node54_tmpdata/xlgeng/ckpt/asr_sot_emotion_caption_task_mix/epoch_1
ckpt_name=step_5999.pt

# 推理for emotion
data_type=shard_full_data # raw  shard_full_data
test_sets="casia cremad esd iemocap_4 meld"
test_data_dir=/home/node54_tmpdata/xlgeng/emotion_data/shards_test/shards_test
gpu_id=3
task=emotion_task # transcribe  sot_task  emotion_task
lang=zh  # en zh
test_sets="${test_sets// /---}"
# shellcheck disable=SC2027
echo "在"$gpu_id"gpu设备上进行"$task"任务的推理,推理数据集为:""$test_sets"
my_named_nohup emotion001 bash decode_common.sh --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task $task --lang $lang