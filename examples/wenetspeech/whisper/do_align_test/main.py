import os
import random

import torch
from gxl_ai_utils.utils import utils_file

from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.init_model import WENET_ENCODER_CLASSES
try:
    import torch_npu
    torch_npu.npu.conv.allow_hf32 = False
    # import deepspeed_npu
    from torch_npu.npu import amp
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    utils_file.logging_warning("torch_npu is not installed, please install torch_npu first if you want to use torch_npu")

from msprobe.pytorch import seed_all,PrecisionDebugger
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

conf_path = "../conf/finetune_whisper_medium_gxl_adapter_multi_task_with_speech_token_huawei_no_dropouut.yaml"
configs = utils_file.load_dict_from_yaml(conf_path)
input_dim = configs['input_dim']

encoder_type = configs.get('encoder', 'conformer')
encoder = WENET_ENCODER_CLASSES[encoder_type](
    input_dim,
    global_cmvn=None,
    **configs['encoder_conf'],
    **configs['encoder_conf']['efficient_conf']
    if 'efficient_conf' in configs['encoder_conf'] else {})

# utils_file.print_model_size(encoder)
# print(encoder)
ckpt_path = "/mnt/sfs/asr/ckpt/epoch_3.pt"
# ckpt_path = "/home/node54_tmpdata/xlgeng/code/wenet_whisper_finetune_xlgeng/examples/wenetspeech/whisper/exp/qwen2_multi_task_4_6gpus_gxl_adapter/epoch_0_with_speech/epoch_3.pt"
checkpoint = torch.load(ckpt_path, map_location='cpu')
new_ckpt = {}
for k, v in checkpoint.items():
    if "encoder." in k:
        new_key = k.split("encoder.")[-1]
        new_ckpt[new_key] = v
missing_keys, unexpected_keys = encoder.load_state_dict(new_ckpt,
                                                      strict=False)
rank = int(os.environ.get('RANK', 0))
# if rank == 0:
#     for key in missing_keys:
#         print("missing tensor: {}".format(key))
#     for key in unexpected_keys:
#         print("unexpected tensor: {}".format(key))
torch.manual_seed(777)
random.seed(777)
utils_file.logging_info('开始严格seed')
seed_all(777)
utils_file.logging_info('结束严格seed')
fake_input_fbank = torch.randn(2, 93, 80)
fake_fbank_lens = torch.tensor([93, 80])
print(f'fake_input_fbank的第一帧的前二十个数值：{fake_input_fbank[0,0, :20]}')
# 吧输入和模型都放在gpu0上
fake_input_fbank = fake_input_fbank.cuda()
fake_fbank_lens = fake_fbank_lens.cuda()
encoder = encoder.cuda()
#
debugger = PrecisionDebugger(config_path='./config_gpu.json', model=encoder)
debugger.start()
output, mask = encoder(fake_input_fbank, fake_fbank_lens)
debugger.stop()
debugger.step()
print(output.shape)
print(f'output的第一帧的前二十个数值：{output[0,0, :20]}')

