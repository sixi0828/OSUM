#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Automatically detect number of gpus
if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi -L | wc -l)
  gpu_list=$(seq -s, 0 $((num_gpus-1)))
else
  num_gpus=-1
  gpu_list="-1"
fi
# You can also manually specify CUDA_VISIBLE_DEVICES
# if you don't want to utilize all available GPU resources.
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"

cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-""}
if [ -z "$cuda_visible_devices" ]; then
  echo "CUDA_VISIBLE_DEVICES is not set. Using default device_ids."
  device_ids=(1)
else
  IFS=',' read -r -a device_ids <<< "$cuda_visible_devices"
  echo "Using CUDA_VISIBLE_DEVICES: $cuda_visible_devices"
fi
# shellcheck disable=SC2145
echo "Parsed device_ids: ${device_ids[@]}"

stage=1
stop_stage=1
# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1
job_id=2024

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=shard_full_data


#test_sets1="test_net_1 test_net_2 test_meeting aishell2 aishell1 speechio_0 speechio_1 speechio_2 speechio_3 speechio_4 speechio_5 speechio_6 speechio_7 speechio_8 speechio_9 speechio_10 speechio_11 speechio_12 speechio_13 speechio_14 speechio_15 speechio_16 speechio_17 speechio_18 speechio_19 speechio_20 speechio_21 speechio_22 speechio_23 speechio_24 speechio_25 speechio_26 librispeech_clean librispeech_other"
test_sets1="test_net_1 test_net_2 test_meeting aishell2 aishell1"
test_sets1=""
test_sets2="aishell4 alimeeting ami"
test_sets2="alimeeting"
english_sets='ami librispeech_clean librispeech_other'

train_config=conf/finetune_whisper_medium.yaml
exp_path=/home/node54_tmpdata/xlgeng/ckpt
#mkdir -p ./exp
#ln -s $exp_path ./exp
dir=$exp_path/asr_sot_task_mix/epoch_2
mkdir -p $dir
tensorboard_dir=$dir/tensorboard
num_workers=8
prefetch=10

checkpoint=$exp_path/asr_sot_task_mix/epoch_1/step_7499.pt

# use average_checkpoint will get better result
average_checkpoint=false
decode_checkpoint=$dir/final.pt
average_num=5
average_mode=step
max_step=88888888
decode_modes="attention"
decoding_chunk_size=-1
ctc_weight=0.5
reverse_weight=0.0
blank_penalty=0.0
length_penalty=0.0
decode_batch=10

train_engine=torch_ddp

# model+optimizer or model_only, model+optimizer is more time-efficient but
# consumes more space, while model_only is the opposite
deepspeed_config=conf/ds_stage1.json
deepspeed_save_states="model+optimizer"
#gxl_data_json_info_path=conf/data.yaml
#python gxl_utils/handle_data_for_weight.py $gxl_data_json_info_path data/tmp/tmp.list

. tools/parse_options.sh || exit 1;


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

  #/home/node54_tmpdata/xlgeng/ckpt/asr_sot_task_mix/epoch_2/aishell1_step_7999.pt_chunk-1_ctc0.5_reverse0.0_blankpenalty0.0_lengthpenalty0.0/attention/text
  for mode in ${decode_modes}; do
      python tools/compute-wer.py --char=1 --v=1 \
        /home/work_nfs8/asr_data/data/asr_test_sets/test_net_1/text /home/node54_tmpdata/xlgeng/ckpt/asr_sot_task_mix/epoch_2/test_net_1_step_7999.pt_chunk-1_ctc0.5_reverse0.0_blankpenalty0.0_lengthpenalty0.0/attention/text > /home/node54_tmpdata/xlgeng/ckpt/asr_sot_task_mix/epoch_2/test_net_1_step_7999.pt_chunk-1_ctc0.5_reverse0.0_blankpenalty0.0_lengthpenalty0.0/attention/wer
  done
  for mode in ${decode_modes}; do
      python tools/compute-wer.py --char=1 --v=1 \
        /home/work_nfs8/asr_data/data/asr_test_sets/test_net_2/text /home/node54_tmpdata/xlgeng/ckpt/asr_sot_task_mix/epoch_2/test_net_2_step_7999.pt_chunk-1_ctc0.5_reverse0.0_blankpenalty0.0_lengthpenalty0.0/attention/text > /home/node54_tmpdata/xlgeng/ckpt/asr_sot_task_mix/epoch_2/test_net_2_step_7999.pt_chunk-1_ctc0.5_reverse0.0_blankpenalty0.0_lengthpenalty0.0/attention/wer
  done
#  for mode in ${decode_modes}; do
#      python tools/compute-wer.py --char=1 --v=1 \
#        /home/work_nfs8/asr_data/data/asr_test_sets/alimeeting/text /home/node54_tmpdata/xlgeng/ckpt/asr_sot_task_mix/epoch_2/alimeeting_step_7999.pt_chunk-1_ctc0.5_reverse0.0_blankpenalty0.0_lengthpenalty0.0/attention/text > /home/node54_tmpdata/xlgeng/ckpt/asr_sot_task_mix/epoch_2/alimeeting_step_7999.pt_chunk-1_ctc0.5_reverse0.0_blankpenalty0.0_lengthpenalty0.0/attention/wer
#  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg${average_num}_mode${average_mode}_max${max_step}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num ${average_num} \
      --mode ${average_mode} \
      --max_step ${max_step} \
      --val_best
  fi
  # Please specify decoding_chunk_size for unified streaming and
  # non-streaming model. The default value is -1, which is full chunk
  # for non-streaming inference.
  decode_checkpoint=/home/node54_tmpdata/xlgeng/ckpt/asr_sot_task_mix/epoch_2/step_7999.pt
  i=0
  # shellcheck disable=SC2154
  for testset in ${test_sets1} ${test_sets2}; do
  {
#    data_type=shard_full_data
    # shellcheck disable=SC2076
    if [[ " ${test_sets2} " =~ " ${testset} " ]]; then
      data_type=shard_full_data
      test_data=/home/work_nfs8/asr_data/data/asr_test_sets/$testset/shards_list.txt
      test_data_dir=/home/work_nfs8/asr_data/data/asr_test_sets/$testset
      task=sot_task
      # 这里可以添加其他操作
    else
      # 当 testset 不属于 test_sets1 或 test_sets2 时执行的操作
      data_type=raw
      task=transcribe
      test_data=/home/work_nfs15/asr_data/data/asr_test_sets/$testset/data.list
      test_data_dir=/home/work_nfs15/asr_data/data/asr_test_sets/$testset
      # 这里可以添加其他操作
    fi

    if [[ " ${english_sets} " =~ " ${testset} " ]]; then
      lang="en"
    else
      lang="zh"
    fi

    base=$(basename $decode_checkpoint)
    result_dir=$dir/${testset}_${base}_chunk${decoding_chunk_size}_ctc${ctc_weight}_reverse${reverse_weight}_blankpenalty${blank_penalty}_lengthpenalty${length_penalty}
    mkdir -p ${result_dir}

    base=$(basename $decode_checkpoint)
    result_dir=$dir/${testset}_${base}_chunk${decoding_chunk_size}_ctc${ctc_weight}_reverse${reverse_weight}_blankpenalty${blank_penalty}_lengthpenalty${length_penalty}
    mkdir -p ${result_dir}
    for mode in ${decode_modes}; do
      python tools/compute-wer.py --char=1 --v=1 \
        $test_data_dir/text $result_dir/$mode/text > $result_dir/$mode/wer
    done
#    device_id=${device_ids[i % ${#device_ids[@]}]}
#    device_id=6
#    echo "Testing ${testset} on GPU ${device_id},lang:"$lang" task:"$task" data_type:"$data_type
#    export CUDA_VISIBLE_DEVICES=$device_id
#    python wenet/bin/recognize.py --gpu ${device_id} \
#      --modes $decode_modes \
#      --config $dir/train.yaml \
#      --data_type $data_type \
#      --test_data "$test_data" \
#      --checkpoint $decode_checkpoint \
#      --beam_size 10 \
#      --batch_size 2 \
#      --blank_penalty ${blank_penalty} \
#      --length_penalty ${length_penalty} \
#      --ctc_weight $ctc_weight \
#      --reverse_weight $reverse_weight \
#      --result_dir $result_dir \
#      --task $task \
#      --lang $lang \
#      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}

#    ((i++))
#    if [[ $device_id -eq $((8 - 1)) ]]; then
#      echo "Waiting for all subprocesses done...device_id为"$device_id
#      wait
#    fi
  }
  done
  wait
#  for testset in ${test_sets1} ${test_sets2}; do
#  {
#    base=$(basename $decode_checkpoint)
#    result_dir=$dir/${testset}_${base}_chunk${decoding_chunk_size}_ctc${ctc_weight}_reverse${reverse_weight}_blankpenalty${blank_penalty}_lengthpenalty${length_penalty}
#    mkdir -p ${result_dir}
#    for mode in ${decode_modes}; do
#      python tools/compute-wer.py --char=1 --v=1 \
#        $test_data_dir/text $result_dir/$mode/text > $result_dir/$mode/wer
#    done
#  }
#  done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Export the best model you want
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip \
    --output_quant_file $dir/final_quant.zip
fi
