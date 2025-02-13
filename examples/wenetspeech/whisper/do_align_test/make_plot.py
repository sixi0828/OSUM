import re


# sentence = "DEBUG TRAIN | steps/sec 0.000| Batch 0/1 loss 2.463887 lr 6.2500e-09 grad_norm 11.363652 rank 0"
def get_info(sentence):
    result_dict = {}
    # 提取每步运行时间（修改后的正则表达式）
    match_steps_sec = re.search(r"steps/sec (\d+\.\d+)", sentence)
    if match_steps_sec:
        result_dict["每秒运行步数"] = float(match_steps_sec.group(1))
    else:
        result_dict["每秒运行步数"] = None

    # 提取batch_index
    match_batch_index = re.search(r"Batch \d+/(\d+)", sentence)
    if match_batch_index:
        result_dict["batch_index"] = int(match_batch_index.group(1))

    # 提取loss
    match_loss = re.search(r"loss (\S+)", sentence)
    if match_loss:
        result_dict["loss"] = float(match_loss.group(1))

    # 提取lr
    match_lr = re.search(r"lr (\S+)", sentence)
    if match_lr:
        result_dict["lr"] = float(match_lr.group(1))

    # 提取grad_norm
    match_grad_norm = re.search(r"grad_norm (\S+)", sentence)
    if match_grad_norm:
        result_dict["grad_norm"] = float(match_grad_norm.group(1))

    return result_dict

input_text_path = "/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/output/run_res/rank0.list"
res_list = []
for line in open(input_text_path, "r", encoding="utf-8"):
    res_list.append(get_info(line))

import re
import pandas as pd

# 将提取的数据整理成DataFrame格式
data = pd.DataFrame(res_list)

# 将DataFrame数据保存为csv文件，这里设置编码为utf-8，不保存索引列
data.to_csv("result.csv", encoding="utf-8", index=False)