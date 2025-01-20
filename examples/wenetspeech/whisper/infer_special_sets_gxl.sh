#source /home/work_nfs11/code/xlgeng/environment/miniconda/etc/profile.d/conda.sh
source ~/.bashrc
#conda activate gxl_base

# dir=exp/qwen2_asr_6gpus_gxl_adapter_init_asr-sot_whisper
dir=/home/node54_tmpdata/yzli/wenet_whisper_finetune/examples/wenetspeech/whisper/exp/qwen2_multi_task_6gpus_gxl_adapter_init_asr-sot_whisper
dir=/home/node54_tmpdata/yzli/wenet_whisper_finetune/examples/wenetspeech/whisper/exp/qwen2_multi_task_2_6gpus_gxl_adapter_init_asr-sot_whisper
dir=/home/node54_tmpdata/xlgeng/ckpt/wenet_whisper_finetune/qwen2_multi_task_3_6gpus_gxl_adapter
dir=/home/node54_tmpdata/xlgeng/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/update_data/epoch_0_with_token
dir=./exp
dir=/mnt/sfs/asr/ckpt/qwen2_multi_task_4_6gpus_gxl_adapter/epoch_1_with_speech_gxl
dir=/home/node54_tmpdata/xlgeng/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/update_data/epoch_1_with_token
dir=/home/node54_tmpdata2/xlgeng/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/update_data/epoch_1_for_emoiton_desc

test_data_dir=/home/work_nfs15/asr_data/data/test_sets_format_3000
test_data_dir=/home/work_nfs16/zxzhao/workspace

ckpt_name=epoch_12.pt

data_type="raw" # "shard_full_data"
lang=zh  # en zh
prompt_file=conf/prompt_stage4.yaml
test_sets="SSL_LLM"
test_sets="${test_sets// /---}"
gpu_id=0
my_named_nohup special_${gpu_id}_${ckpt_name}_new bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --lang $lang --prompt_file $prompt_file
#
#
#test_sets="public_test/kaggle_gender public_test/kaggle_age"
#test_sets="${test_sets// /---}"
#gpu_id=1
#my_named_nohup special_${gpu_id}_${ckpt_name} bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --lang $lang --prompt_file $prompt_file
#
#test_sets="public_test/aishell1_gender public_test/roobo_100"
#test_sets="${test_sets// /---}"
#gpu_id=2
#my_named_nohup special_${gpu_id}_${ckpt_name} bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --lang $lang --prompt_file $prompt_file
#
#
#test_sets="gender"
#test_sets="${test_sets// /---}"
#gpu_id=3
#my_named_nohup special_${gpu_id}_${ckpt_name}_next bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --lang $lang --prompt_file $prompt_file
#
#
#test_sets="age public_test/AirBench_speech chat"
#test_sets="${test_sets// /---}"
#gpu_id=4
#my_named_nohup special_${gpu_id}_${ckpt_name}_next bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --lang $lang --prompt_file $prompt_file
#
#test_sets="style"
#test_sets="${test_sets// /---}"
#gpu_id=5
#my_named_nohup special_${gpu_id}_${ckpt_name}_next bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --lang $lang --prompt_file $prompt_file
#
#
#
#test_sets="emotion"
#test_sets="${test_sets// /---}"
#gpu_id=6
#my_named_nohup special_${gpu_id}_${ckpt_name}_next bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --lang $lang --prompt_file $prompt_file
#
#test_sets="public_test/MER23_test public_test/MELD_test"
#test_sets="${test_sets// /---}"
#gpu_id=7
#my_named_nohup special_${gpu_id}_${ckpt_name} bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --lang $lang --prompt_file $prompt_file
