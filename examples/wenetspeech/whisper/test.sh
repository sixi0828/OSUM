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
export CUDA_VISIBLE_DEVICES="3,7"
#export CUDA_VISIBLE_DEVICES="2"
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

stage=0
stop_stage=0

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1
job_id=2024

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=shard_full_data

train_set=train_l
dev_set=dev
test_sets="test_net_1 test_net_2 test_meeting"
test_sets="aishell4 alimeeting ami"

train_config=conf/test.yaml
exp_path=exp
dir=$exp_path/test1
mkdir -p $dir
tensorboard_dir=$dir/tensorboard
num_workers=8
prefetch=10

checkpoint=exp/whisper-medium/wenet_whisper.pt
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

train_engine=deepspeed # torch_ddp

# model+optimizer or model_only, model+optimizer is more time-efficient but
# consumes more space, while model_only is the opposite
deepspeed_config=conf/ds_stage2.json
deepspeed_save_states="model+optimizer"
gxl_data_json_info_path=conf/data.yaml

cv_data=data/asr_cv.list
#cv_data=/home/node54_tmpdata/xlgeng/emotion_data/shards_dev/shards_dev/shards_list.txt

. tools/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  mkdir -p $dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  # NOTE(xcsong): deepspeed fails with gloo, see
  #   https://github.com/microsoft/DeepSpeed/issues/2818
  dist_backend="nccl"

  # train.py rewrite $train_config to $dir/train.yaml with model input
  # and output dimension, and $dir/train.yaml will be used for inference
  # and export.
  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi

  # NOTE(xcsong): Both ddp & deepspeed can be launched by torchrun
  # NOTE(xcsong): To unify single-node & multi-node training, we add
  #               all related args. You should change `nnodes` &
  #               `rdzv_endpoint` for multi-node, see
  #               https://pytorch.org/docs/stable/elastic/run.html#usage
  #               https://github.com/wenet-e2e/wenet/pull/2055#issuecomment-1766055406
  #               `rdzv_id` - A user-defined id that uniquely identifies the worker group for a job.
  #                           This id is used by each node to join as a member of a particular worker group.
  #               `rdzv_endpoint` - The rendezvous backend endpoint; usually in form <host>:<port>.
  # NOTE(xcsong): In multi-node training, some clusters require special NCCL variables to set prior to training.
  #               For example: `NCCL_IB_DISABLE=1` + `NCCL_SOCKET_IFNAME=enp` + `NCCL_DEBUG=INFO`
  #               without NCCL_IB_DISABLE=1
  #                   RuntimeError: NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1269, internal error, NCCL Version xxx
  #               without NCCL_SOCKET_IFNAME=enp  (IFNAME could be get by `ifconfig`)
  #                   RuntimeError: The server socket has failed to listen on any local network address. The server socket has failed to bind to [::]:xxx
  #               ref: https://github.com/google/jax/issues/13559#issuecomment-1343573764
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wenet/bin/train.py \
      --train_engine ${train_engine} \
      --config $train_config \
      --data_type  $data_type \
      --train_data data/tmp/1 \
      --cv_data data/tmp/1 \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --tensorboard_dir ${tensorboard_dir} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --timeout 1200 \
      --use_amp \
      --deepspeed_config ${deepspeed_config} \
      --deepspeed.save_states ${deepspeed_save_states} >> $dir/log 2>&1
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  if [ "$deepspeed_save_states" = "model+optimizer" ]; then
    subdir=$dir/step_22499
    tag=$(basename "$subdir")
    echo "$tag"
    python3 ${dir}/zero_to_fp32.py \
      ${dir} ${dir}/${tag}.pt -t ${tag}
    rm -rf ${dir}/${tag}
    # for subdir in $(find "$dir" -maxdepth 1 -type d | grep -v "^$dir$")
    # do
    #   if [ $(find "$subdir" -mindepth 1 -type d | wc -l) -eq 0 ]; then
    #     # NOTE(xcsong): zero_to_fp32.py is automatically generated by deepspeed
    #     tag=$(basename "$subdir")
    #     echo "$tag"
    #     python3 ${dir}/zero_to_fp32.py \
    #       ${dir} ${dir}/${tag}.pt -t ${tag}
    #     rm -rf ${dir}/${tag}
    #   fi
    # done
  fi
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
  decode_checkpoint=$dir/step_499.pt
  i=0
  for testset in ${test_sets}; do
  {
    base=$(basename $decode_checkpoint)
    result_dir=$dir/${testset}_${base}_chunk${decoding_chunk_size}_ctc${ctc_weight}_reverse${reverse_weight}_blankpenalty${blank_penalty}_lengthpenalty${length_penalty}
    mkdir -p ${result_dir}
    device_id=${device_ids[i % ${#device_ids[@]}]}
    echo "Testing ${testset} on GPU ${device_id}"
    export CUDA_VISIBLE_DEVICES=$device_id
    python wenet/bin/recognize.py --gpu ${device_id} \
      --modes $decode_modes \
      --config $dir/train.yaml \
      --data_type shard_full_data \
      --test_data /home/work_nfs8/xlgeng/data/scp_test/$testset/shards_list.txt \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size ${decode_batch} \
      --blank_penalty ${blank_penalty} \
      --length_penalty ${length_penalty} \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      --result_dir $result_dir \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size} &
    ((i++))
    if [[ $device_id -eq $((num_gpu - 1)) ]]; then
      wait
    fi
  }
  done
  wait
  for testset in ${test_sets}; do
  {
    base=$(basename $decode_checkpoint)
    result_dir=$dir/${testset}_${base}_chunk${decoding_chunk_size}_ctc${ctc_weight}_reverse${reverse_weight}_blankpenalty${blank_penalty}_lengthpenalty${length_penalty}
    mkdir -p ${result_dir}
    for mode in ${decode_modes}; do
      python tools/compute-wer.py --char=1 --v=1 \
        /home/work_nfs8/xlgeng/data/scp_test/$testset/text $result_dir/$mode/text > $result_dir/$mode/wer
    done
  }
  done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Export the best model you want
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip \
    --output_quant_file $dir/final_quant.zip
fi
