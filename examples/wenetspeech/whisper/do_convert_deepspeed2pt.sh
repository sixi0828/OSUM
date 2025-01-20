dir="/home/xlgeng/node54_tmpdata/xlgeng/code/wenet_whisper_finetune/examples/wenetspeech/whisper/exp/qwen2_multi_task_2_6gpus_gxl_adapter_init_asr-sot_whisper"
dir=/home/node54_tmpdata/xlgeng/ckpt/wenet_whisper_finetune/qwen2_multi_task_3_6gpus_gxl_adapter
dir=/home/node54_tmpdata/xlgeng/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/update_data/epoch_1_with_token
tag='epoch_5'
python3 ${dir}/zero_to_fp32.py \
    ${dir} ${dir}/${tag}.pt -t ${tag}
rm -rf ${dir}/${tag}