#source /home/work_nfs11/code/xlgeng/environment/miniconda/etc/profile.d/conda.sh
source /home/xlgeng/.bashrc
#conda activate gxl_base

# dir=exp/qwen2_asr_6gpus_gxl_adapter_init_asr-sot_whisper
dir=/home/node54_tmpdata/yzli/wenet_whisper_finetune/examples/wenetspeech/whisper/exp/qwen2_multi_task_6gpus_gxl_adapter_init_asr-sot_whisper
dir=/home/node54_tmpdata/yzli/wenet_whisper_finetune/examples/wenetspeech/whisper/exp/qwen2_multi_task_2_6gpus_gxl_adapter_init_asr-sot_whisper
dir=/home/node54_tmpdata/xlgeng/ckpt/wenet_whisper_finetune/qwen2_multi_task_3_6gpus_gxl_adapter
dir=/home/node54_tmpdata/xlgeng/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/update_data/epoch_0_with_token


#test_data_dir="/home/node54_tmpdata/xlgeng/code/wenet_whisper_finetune_xlgeng/examples/wenetspeech/whisper/data/test_sets"
test_data_dir='/home/node54_tmpdata/xlgeng/code/wenet_whisper_finetune_xlgeng/examples/wenetspeech/whisper/data/test_sets_format_3000'
ckpt_name=epoch_0.pt
data_type="raw" # "shard_full_data"
lang=zh  # en zh
prompt_file=conf/prompt_stage4.yaml
task=special_text_task


test_sets="librispeech_clean test_net_1 aishell speechio_0 speechio_1 speechio_3"
test_sets="${test_sets// /---}"
gpu_id=5
# prompt_file=conf/prompt_stage2.yaml
my_named_nohup ${task}_${gpu_id}_${ckpt_name} bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file


task="<TRANSCRIBE>"
#task="<SOT>"
#data_type="shard_full_data" # "shard_full_data"
test_sets="aishell2 test_net_2 speechio_2 speechio_4"
#test_sets="ami"
test_sets="${test_sets// /---}"
gpu_id=6
# prompt_file=conf/prompt_stage2.yaml
my_named_nohup ${task}_${gpu_id}_${ckpt_name} bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file


task="<TRANSCRIBE>"
#task="<SOT>"
#data_type="shard_full_data" # "shard_full_data"
test_sets="librispeech_other test_meeting"
#test_sets="alimeeting"
test_sets="${test_sets// /---}"
gpu_id=7
 prompt_file=conf/prompt_stage2.yaml
my_named_nohup ${task}_${gpu_id}_${ckpt_name} bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file
