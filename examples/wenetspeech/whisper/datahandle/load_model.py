from transformers import AutoTokenizer, AutoModelForCausalLM
from gxl_ai_utils.utils import utils_file
import os
# os.environ['HF_ENDPOINT']="https://hf-mirror.com"  # 在命令行里面加入就可以了，别的代码完全不用动
# export HF_ENDPOINT=https://hf-mirror.com
#export HF_HOME=/mnt/sfs/asr/ckpt
#export TRANSFORMERS_CACHE=/mnt/sfs/asr/ckpt

# 加载 tokenizer 和模型，并指定 cache_dir 保存模型文件
# model_path = "/home/work_nfs15/asr_data/ckpt/Phi-3.5-mini-instruct/models--microsoft--Phi-3.5-mini-instruct/snapshots/af0dfb8029e8a74545d0736d30cb6b58d2f0f3f0"
model_path = "/home/node54_tmpdata/xlgeng/ckpt/qwen-7B-instruct/qwen2_7b"
model_path="Qwen/Qwen2-7B"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,)
# vocab_dict = tokenizer.get_vocab()
# utils_file.write_dict_to_scp(vocab_dict, "./vocab_dict.scp")
print(model)