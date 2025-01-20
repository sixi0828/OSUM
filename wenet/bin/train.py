# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import datetime
import logging
import os
import random

import numpy as np
import yaml
import torch

import torch.distributed as dist

from torch.distributed.elastic.multiprocessing.errors import record
from wenet.utils.common import lrs_to_str, TORCH_NPU_AVAILABLE  # noqa just ensure to check torch-npu

from wenet.utils.executor import Executor
from wenet.utils.config import override_config
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.train_utils import (
    add_fsdp_args, add_model_args, add_dataset_args, add_ddp_args,
    add_deepspeed_args, add_trace_args, init_distributed,
    init_dataset_and_dataloader, check_modify_and_save_config,
    init_optimizer_and_scheduler, init_scaler, trace_and_print_model,
    wrap_cuda_model, init_summarywriter, save_model, log_per_epoch,
    add_lora_args, reinit_lora)
from gxl_ai_utils.utils import utils_file

try:
    import torch_npu

    torch_npu.npu.conv.allow_hf32 = False
    # import deepspeed_npu
    from torch_npu.npu import amp
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    utils_file.logging_warning(
        "torch_npu is not installed, please install torch_npu first if you want to use torch_npu")
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

from msprobe.pytorch import seed_all
import gc

gc.set_threshold(700, 10, 10000)  # python gc阈值设置


# import deepspeed_npu
def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'torch_fsdp', 'deepspeed'],
                        help='Engine for paralleled training')
    # set default value of device to "cuda", avoiding the modify of original scripts
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        choices=["cpu", "npu", "cuda"],
                        help='accelerator for training')
    # load deepspeed checkpoint
    parser.add_argument('--load_dir',
                        type=str,
                        default=None)
    parser.add_argument('--ckpt_id',
                        type=str,
                        default=None)
    parser = add_model_args(parser)
    parser = add_dataset_args(parser)
    parser = add_ddp_args(parser)
    parser = add_lora_args(parser)
    parser = add_deepspeed_args(parser)
    parser = add_fsdp_args(parser)
    parser = add_trace_args(parser)
    args = parser.parse_args()
    if args.train_engine == "deepspeed":
        args.deepspeed = True
        assert args.deepspeed_config is not None
    return args


# NOTE(xcsong): On worker errors, this recod tool will summarize the
#   details of the error (e.g. time, rank, host, pid, traceback, etc).
@record
def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    # Set random seed
    torch.manual_seed(777)
    random.seed(777)
    np.random.seed(777)
    utils_file.logging_info('开始严格seed')
    seed_all(777)
    utils_file.logging_info('结束严格seed')
    logging.info('Random seed set to {}'.format(777))

    # Read config
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    # init tokenizer
    tokenizer = init_tokenizer(configs)

    # Init env for ddp OR deepspeed
    _, _, rank = init_distributed(args)

    # Init asr model from configs
    model, configs = init_model(args, configs)

    # Get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, tokenizer)

    # Do some sanity checks and save config to arsg.model_dir
    configs = check_modify_and_save_config(args, configs,
                                           tokenizer.symbol_table)

    if hasattr(args, 'lora_reinit') and args.lora_reinit:
        reinit_lora(model, args, configs, tokenizer)

    # Check model is jitable & print model archtectures
    trace_and_print_model(args, model)

    # Tensorboard summary
    writer = init_summarywriter(args)

    # Dispatch model from cpu to gpu
    model, device = wrap_cuda_model(args, model, configs)

    # Get optimizer & scheduler
    model, optimizer, scheduler = init_optimizer_and_scheduler(
        args, configs, model)

    # Load deepspeed checkpoint
    if args.load_dir is not None and \
            args.ckpt_id is not None:
        _, client_sd = model.load_checkpoint(args.load_dir, args.ckpt_id)

    # Save checkpoints
    # save_model(model,
    #            info_dict={
    #                "save_time":
    #                datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
    #                "tag":
    #                "init",
    #                **configs
    #            })

    # Get executor
    tag = configs["init_infos"].get("tag", "init")
    executor = Executor(global_step=configs["init_infos"].get('step', -1),
                        device=device)

    # Init scaler, used for pytorch amp mixed precision training
    scaler = init_scaler(args)

    # Start training loop
    start_epoch = configs["init_infos"].get('epoch', 0) + int("epoch_" in tag)
    # if save_interval in configs, steps mode else epoch mode
    end_epoch = configs.get('max_epoch', 100)
    assert start_epoch <= end_epoch
    configs.pop("init_infos", None)
    final_epoch = None
    for epoch in range(start_epoch, end_epoch):
        configs['epoch'] = epoch

        lrs = [group['lr'] for group in optimizer.param_groups]
        logging.info('Epoch {} Step {} TRAIN info lr {} rank {}'.format(
            epoch, executor.step, lrs_to_str(lrs), rank))

        dist.barrier(
        )  # NOTE(xcsong): Ensure all ranks start Train at the same time.
        # NOTE(xcsong): Why we need a new group?  see `train_utils.py::wenet_join`
        group_join = dist.new_group(  # fix by zhaoyi for 多机训练
            backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        # group_join = None
        executor.train(model, optimizer, scheduler, train_data_loader,
                       cv_data_loader, writer, configs, scaler, group_join)
        # dist.destroy_process_group(group_join)

        dist.barrier(
        )  # NOTE(xcsong): Ensure all ranks start CV at the same time.
        loss_dict = executor.cv(model, cv_data_loader, configs)
        info_dict = {
            'epoch': epoch,
            'lrs': [group['lr'] for group in optimizer.param_groups],
            'step': executor.step,
            "loss_dict": loss_dict,
            'save_time': datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            'tag': "epoch_{}".format(epoch),
            'loss_dict': loss_dict,
            **configs
        }
        # epoch cv: tensorboard && log
        log_per_epoch(writer, info_dict=info_dict)
        save_model(model, info_dict=info_dict)

        final_epoch = epoch

    if final_epoch is not None and rank == 0:
        final_model_path = os.path.join(args.model_dir, 'final.pt')
        os.remove(final_model_path) if os.path.exists(
            final_model_path) else None
        os.symlink('{}.pt'.format(final_epoch), final_model_path)
        writer.close()
    dist.barrier(
    )  # NOTE(yktian): Ensure all ranks end Train before destroy process group.
    dist.destroy_process_group()


if __name__ == '__main__':
    main()