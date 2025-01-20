# 激活conda环境（如果需要的话，这里假设gxl_base是你的环境名，可根据实际调整）
# conda activate gxl_base
source ~/.bashrc
cd ..

# 定义测试数据目录，可根据实际情况修改路径
test_data_dir='/home/node54_tmpdata/xlgeng/code/wenet_whisper_finetune_xlgeng/examples/wenetspeech/whisper/data/test_sets_format_3000'
# 定义语言，可选值为en或zh
lang=zh
# 获取命令行传入的第一个参数作为gpu_id
gpu_id=$1
prompt_file=conf/prompt_stage3.yaml
echo 'gpu_id: '$gpu_id

dirs=(
    "/home/node54_tmpdata/xlgeng/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp"
#    "/home/node54_tmpdata/xlgeng/ckpt/wenet_whisper_finetune/qwen2_multi_task_3_6gpus_gxl_adapter"
#    "/home/node54_tmpdata/xlgeng/ckpt/wenet_whisper_finetune/qwen2_multi_task_3_6gpus_gxl_adapter"
#    "/home/node54_tmpdata/yzli/wenet_whisper_finetune/examples/wenetspeech/whisper/exp/qwen2_multi_task_2_6gpus_gxl_adapter_init_asr-sot_whisper"
)

# 定义不同的检查点名称列表，与dirs列表中的元素顺序对应，可根据实际调整对应关系
ckpt_names=(
    "epoch_0.pt"
)

# 定义不同的任务类型列表
tasks=(
    "<TRANSCRIBE><ALIGN>"
)

# 定义不同的测试集名称列表，与tasks列表中的元素顺序对应，可根据实际调整对应关系
test_sets=(
    "align"
)

# 获取任务列表的长度，即索引的最大值
task_length=${#tasks[@]}

# 多重for循环，嵌套遍历不同维度的参数组合
for dir in "${dirs[@]}"; do
    for ((i = 0; i < ${#ckpt_names[@]}; i++)); do
        for ((index = 0; index < task_length; index++)); do
            task=${tasks[index]}
            test_set=${test_sets[index]}
            data_type='raw'
            test_sets_mod="${test_set// /---}"
            echo "task: $task, test_set: $test_set"
            bash gxl_decode/decode_common_gxl.sh  --data_type $data_type --test_sets "$test_sets_mod" --test_data_dir $test_data_dir --gpu_id $gpu_id --dir $dir --ckpt_name "${ckpt_names[$i]}" --task "$task" --lang $lang --prompt_file $prompt_file
        done
    done
done