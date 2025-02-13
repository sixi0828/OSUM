#source /home/work_nfs11/code/xlgeng/environment/miniconda/etc/profile.d/conda.sh
source /mnt/sfs/asr/.bashrc
#conda activate gxl_base


#test_data_dir="/home/node54_tmpdata/xlgeng/code/wenet_whisper_finetune_xlgeng/examples/wenetspeech/whisper/data/test_sets"
test_data_dir="/mnt/sfs/asr/test_data/asr_test_sets"



dir=/mnt/sfs/asr/ckpt/qwen2_multi_task_4_6gpus_gxl_adapter/epoch_5_with_speech_gxl
ckpt_name=step_21249.pt

dir="/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/epoch_6_with_speech_gxl_new_with_asr-chat"
ckpt_name=step_19999.pt

dir="/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/epoch_6_with_speech_gxl"
ckpt_name="step_76249.pt"

dir="/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/epoch_10_11_with_speech_gxl_with_asr-chat"
ckpt_name="step_51249.pt"
dir="/mnt/sfs/asr/ckpt/qwen2-7B-instruct_multi_task_4_gxl_adapter/epoch_0"
ckpt_name=step_24999.pt
dir=/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/epoch_13_with_speech_gxl_with_asr-chat_full-data
ckpt_name=step_37499.pt
dir="/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/epoch_13_with_asr-chat_full_data"
ckpt_name="step_32499.pt"

data_type="raw" # "shard_full_data"
lang=zh  # en zh
prompt_file=conf/prompt_stage4.yaml

task="<TRANSCRIBE>"
#task="<SOT>"
#data_type="shard_full_data" # "shard_full_data"
test_sets="test_net_1"
#test_sets="aishell4"
test_sets="${test_sets// /---}"
gpu_id=0
# prompt_file=conf/prompt_stage2.yaml
my_named_nohup ${task}_${gpu_id}_${ckpt_name} bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file


task="<TRANSCRIBE>"
#task="<SOT>"
#data_type="shard_full_data" # "shard_full_data"
test_sets="test_net_2"
#test_sets="ami"
test_sets="${test_sets// /---}"
gpu_id=1
# prompt_file=conf/prompt_stage2.yaml
my_named_nohup ${task}_${gpu_id}_${ckpt_name} bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file


task="<TRANSCRIBE>"
#task="<SOT>"
#data_type="shard_full_data" # "shard_full_data"
test_sets="test_meeting"
#test_sets="alimeeting"
test_sets="${test_sets// /---}"
gpu_id=2
 prompt_file=conf/prompt_stage2.yaml
my_named_nohup ${task}_${gpu_id}_${ckpt_name} bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file

task="<TRANSCRIBE>"
#task="<SOT>"
#data_type="shard_full_data" # "shard_full_data"
test_sets="aishell2 speechio_0 speechio_1 speechio_3"
#test_sets="aishell4"
test_sets="${test_sets// /---}"
gpu_id=3
# prompt_file=conf/prompt_stage2.yaml
my_named_nohup ${task}_${gpu_id}_${ckpt_name} bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file

task="<TRANSCRIBE>"
#task="<SOT>"
#data_type="shard_full_data" # "shard_full_data"
test_sets="librispeech_clean librispeech_other speechio_4 speechio_2"
#test_sets="aishell4"
test_sets="${test_sets// /---}"
gpu_id=4
# prompt_file=conf/prompt_stage2.yaml
my_named_nohup ${task}_${gpu_id}_${ckpt_name} bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file
