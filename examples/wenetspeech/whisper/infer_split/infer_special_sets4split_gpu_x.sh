source ~/.bashrc

cd ..
dir=/home/node54_tmpdata/yzli/wenet_whisper_finetune/examples/wenetspeech/whisper/exp/qwen2_multi_task_6gpus_gxl_adapter_init_asr-sot_whisper
dir=/home/node54_tmpdata/yzli/wenet_whisper_finetune/examples/wenetspeech/whisper/exp/qwen2_multi_task_2_6gpus_gxl_adapter_init_asr-sot_whisper
dir=/home/node54_tmpdata/xlgeng/ckpt/wenet_whisper_finetune/qwen2_multi_task_3_6gpus_gxl_adapter

# for special set
test_data_dir='/home/node54_tmpdata/xlgeng/code/wenet_whisper_finetune_xlgeng/examples/wenetspeech/whisper/data/test_sets'

ckpt_name='epoch_6.pt'
data_type='raw' # 'shard_full_data'
lang=zh  # en zh
prompt_file=conf/prompt_stage3.yaml



gpu_id=$1

#task='<TRANSCRIBE><ALIGN>'
#data_type='raw' # 'shard_full_data'
#test_sets="align/aishell2/split_8/index_${gpu_id}" # 15000
#test_sets="${test_sets// /---}"
#bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file

task='<TRANSCRIBE><AGE>'
data_type='raw'
test_sets="age/split_8/index_${gpu_id} age/child/split_8/index_${gpu_id}"
test_sets="${test_sets// /---}"
bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file

task='<TRANSCRIBE><EMOTION>'
data_type='raw' # 'shard_full_data'
test_sets="emotion/split_8/index_${gpu_id}" # 4000
test_sets="${test_sets// /---}"
bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file

task='<TRANSCRIBE><GENDER>'
test_sets="gender/split_8/index_${gpu_id}" # 13000
data_type='raw' # 'shard_full_data'
test_sets="${test_sets// /---}"
bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file
