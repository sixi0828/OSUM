import torch
from gxl_ai_utils.utils import utils_file

prompt_dict = utils_file.load_dict_from_yaml('conf/prompt.yaml')
print(prompt_dict)
path = "/home/node54_tmpdata/xlgeng/ckpt/asr_sot_emotion_caption_task_mix/epoch_2/step_20999.pt"
checkpoint = torch.load(path, map_location='cpu')
for key, value in checkpoint.items():
    print(key, value.shape)

