#source /home/work_nfs11/code/xlgeng/environment/miniconda/etc/profile.d/conda.sh
source /mnt/sfs/asr/.bashrc
#conda activate gxl_base


test_data_dir=/mnt/sfs/asr/test_data/test_sets_format_3000

dir=/mnt/sfs/asr/ckpt/qwen2_multi_task_4_6gpus_gxl_adapter/epoch_4_with_speech_gxl_new
ckpt_name=step_24999.pt

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

dir="/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/only_emotion_from_epoch11"
ckpt_name="step_27499.pt"
dir="/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/only_emotion_from_epoch11_with_ssl_vec"
ckpt_name="step_28749.pt"

dir="/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/epoch_13_with_asr-chat_full_data"
ckpt_name="step_32499.pt"

data_type="raw" # "shard_full_data"
lang=zh  # en zh
prompt_file=conf/prompt_stage4.yaml


# # -------------------format ----------------------------
test_sets="caption_0107_esc50" # 1W 条
test_sets="${test_sets// /---}"
gpu_id=0
my_named_nohup special_${gpu_id}_${ckpt_name} bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --lang $lang --prompt_file $prompt_file

test_sets="caption_0107_vocalsound  public_test/roobo_100"
test_sets="${test_sets// /---}"
gpu_id=1
my_named_nohup special_${gpu_id}_${ckpt_name} bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --lang $lang --prompt_file $prompt_file

test_sets="public_test/kaggle_gender public_test/kaggle_age"
test_sets="${test_sets// /---}"
gpu_id=2
my_named_nohup special_${gpu_id}_${ckpt_name} bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --lang $lang --prompt_file $prompt_file

test_sets="public_test/aishell1_gender"
test_sets="${test_sets// /---}"
gpu_id=3
my_named_nohup special_${gpu_id}_${ckpt_name} bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --lang $lang --prompt_file $prompt_file


test_sets="age aslp_chat_test public_test/AirBench_speech chat"
test_sets="${test_sets// /---}"
gpu_id=4
my_named_nohup special_${gpu_id}_${ckpt_name}_next bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --lang $lang --prompt_file $prompt_file

test_sets="style"
test_sets="${test_sets// /---}"
gpu_id=5
my_named_nohup special_${gpu_id}_${ckpt_name}_next bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --lang $lang --prompt_file $prompt_file

test_sets="caption_aslp_record emotion"
test_sets="${test_sets// /---}"
gpu_id=6
my_named_nohup special_${gpu_id}_${ckpt_name}_next bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --lang $lang --prompt_file $prompt_file

test_sets="gender caption_aslp_record public_test/MER23_test public_test/MELD_test"
test_sets="${test_sets// /---}"
gpu_id=7
my_named_nohup special_${gpu_id}_${ckpt_name} bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --lang $lang --prompt_file $prompt_file

test_sets="aligin_english_clean"
test_sets="${test_sets// /---}"
gpu_id=5
my_named_nohup special_${gpu_id}_${ckpt_name}_next bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --lang $lang --prompt_file $prompt_file


test_sets="align_cn_noize"
test_sets="${test_sets// /---}"
gpu_id=6
my_named_nohup special_${gpu_id}_${ckpt_name}_next bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --lang $lang --prompt_file $prompt_file


test_sets="align_en_noize"
test_sets="${test_sets// /---}"
gpu_id=7
my_named_nohup special_${gpu_id}_${ckpt_name}_next bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --lang $lang --prompt_file $prompt_file


# # # ----------------format --------------------------------




# -------------------format emotion only ----------------------------

# test_sets="emotion"
# test_sets="${test_sets// /---}"
# gpu_id=6
# my_named_nohup special_${gpu_id}_${ckpt_name} bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --lang $lang --prompt_file $prompt_file

# test_sets="public_test/MER23_test public_test/MELD_test"
# test_sets="${test_sets// /---}"
# gpu_id=7
# my_named_nohup special_${gpu_id}_${ckpt_name} bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --lang $lang --prompt_file $prompt_file

# # ----------------format emotion only --------------------------------

