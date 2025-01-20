source /home/work_nfs11/code/xlgeng/environment/miniconda/etc/profile.d/conda.sh
conda activate gxl_base

# dir=exp/qwen2_asr_6gpus_gxl_adapter_init_asr-sot_whisper
dir=exp/qwen2_multi_task_6gpus_gxl_adapter_init_asr-sot_whisper

gpu_id=0

# test_sets='aishell1 aishell2'
# test_sets='test_net'
# test_sets='test_meeting'
# test_sets='speechio_1 speechio_2 speechio_3'
# test_sets='speechio_0 speechio_4 speechio_5 speechio_6'
# test_sets='emotion'
# test_sets='pure_background'
# test_sets='add_mid_background'
# test_sets='add_end_background'
# test_sets='chat'
# test_sets='align/aishell2'
# test_sets='age'
test_sets='age/child'
# test_sets='gender'
# test_sets='caption'

# test_data_dir='/home/work_nfs15/asr_data/data/asr_test_sets'
test_data_dir='data/test_sets'

ckpt_name=epoch_9.pt

# task='<TRANSCRIBE><GENDER>'
# task='<TRANSCRIBE><EMOTION>'
# task='<TRANSCRIBE><BACKGROUND>'
# task='<TRANSCRIBE>'
# task='<S2TCHAT>'
# task='<TRANSCRIBE><ALIGN>'
task='<TRANSCRIBE><AGE>'
# task='<TRANSCRIBE><GENDER>'
# task='<TRANSCRIBE><CAPTION>'

data_type='raw' # 'shard_full_data'
lang=zh  # en zh

bash gxl_decode/decode_common.sh \
    --data_type $data_type \
    --test_sets "$test_sets" \
    --test_data_dir $test_data_dir \
    --gpu_id $gpu_id \
    --dir $dir \
    --ckpt_name $ckpt_name \
    --task "$task" \
    --lang $lang
