
#conda activate gxl_base
source ~/.bashrc
cd ..

# dir=exp/qwen2_asr_6gpus_gxl_adapter_init_asr-sot_whisper
# dir=/home/node54_tmpdata/yzli/wenet_whisper_finetune/examples/wenetspeech/whisper/exp/qwen2_multi_task_6gpus_gxl_adapter_init_asr-sot_whisper
# dir=/home/node54_tmpdata/yzli/wenet_whisper_finetune/examples/wenetspeech/whisper/exp/qwen2_multi_task_2_6gpus_gxl_adapter_init_asr-sot_whisper
test_data_dir='/home/node54_tmpdata/xlgeng/code/wenet_whisper_finetune/examples/wenetspeech/whisper/gxl_data/test_sets'

lang=zh  # en zh
gpu_id=$1
prompt_file=conf/prompt_stage3.yaml
echo 'gpu_id: '$gpu_id


dir=/home/node54_tmpdata/xlgeng/ckpt/wenet_whisper_finetune/qwen2_multi_task_3_6gpus_gxl_adapter
ckpt_name=epoch_2.pt


task='<TRANSCRIBE><AGE>'
data_type='raw' # 'shard_full_data'
test_sets='age' # age:13271 child:3000
test_sets="${test_sets// /---}"
bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file


task='<TRANSCRIBE><BACKGROUND>'
data_type='raw' # 'shard_full_data'
test_sets='add_mid_background' #  15000
test_sets="${test_sets// /---}"
bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file




dir=/home/node54_tmpdata/xlgeng/ckpt/wenet_whisper_finetune/qwen2_multi_task_3_6gpus_gxl_adapter
ckpt_name=epoch_6.pt

task='<TRANSCRIBE><AGE>'
data_type='raw' # 'shard_full_data'
test_sets='age' # age:13271 child:3000
test_sets="${test_sets// /---}"
bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file

task='<TRANSCRIBE><BACKGROUND>'
data_type='raw' # 'shard_full_data'
test_sets='add_mid_background' #  15000
test_sets="${test_sets// /---}"
bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file





dir=/home/node54_tmpdata/xlgeng/ckpt/wenet_whisper_finetune/qwen2_multi_task_3_6gpus_gxl_adapter
ckpt_name=epoch_2.pt


task='<TRANSCRIBE><AGE>'
data_type='raw' # 'shard_full_data'
test_sets='age' # age:13271 child:3000
test_sets="${test_sets// /---}"
bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file


task='<TRANSCRIBE><BACKGROUND>'
data_type='raw' # 'shard_full_data'
test_sets='add_mid_background' #  15000
test_sets="${test_sets// /---}"
bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file




dir=/home/node54_tmpdata/yzli/wenet_whisper_finetune/examples/wenetspeech/whisper/exp/qwen2_multi_task_2_6gpus_gxl_adapter_init_asr-sot_whisper
ckpt_name=epoch_9.pt

task='<TRANSCRIBE><AGE>'
data_type='raw' # 'shard_full_data'
test_sets='age' # age:13271 child:3000
test_sets="${test_sets// /---}"
bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file

task='<TRANSCRIBE><BACKGROUND>'
data_type='raw' # 'shard_full_data'
test_sets='add_mid_background' #  15000
test_sets="${test_sets// /---}"
bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file




dir=/home/node54_tmpdata/xlgeng/ckpt/wenet_whisper_finetune/qwen2_multi_task_3_6gpus_gxl_adapter
ckpt_name=epoch_8.pt


task='<TRANSCRIBE><AGE>'
data_type='raw' # 'shard_full_data'
test_sets='age' # age:13271 child:3000
test_sets="${test_sets// /---}"
bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file


task='<TRANSCRIBE><BACKGROUND>'
data_type='raw' # 'shard_full_data'
test_sets='add_mid_background' #  15000
test_sets="${test_sets// /---}"
bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name $ckpt_name --task "$task" --lang $lang --prompt_file $prompt_file
