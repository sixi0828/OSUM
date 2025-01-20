import time

import gradio as gr
import os

import sys

import torch
import torchaudio
from torch import dtype

sys.path.insert(0, '../../../../')
from gxl_ai_utils.utils import utils_file
from wenet.utils.init_tokenizer import init_tokenizer
from gxl_ai_utils.config.gxl_config import GxlNode
from wenet.utils.init_model import init_model
import logging
import librosa
import torch
import torchaudio
import numpy as np




def remove_silence_torchaudio_ends(input_path, output_path, threshold=0.01, silence_duration_threshold=0.03):
    # 读取音频文件
    waveform, sr = torchaudio.load(input_path)
    # 计算音频的能量
    energy = torch.sum(waveform ** 2, dim=0)
    # 找到静音部分的索引
    is_silent = energy < threshold
    silent_indices = torch.where(is_silent)[0]
    # 计算静音部分的持续时间
    silent_durations = []
    start = silent_indices[0]
    for i in range(1, len(silent_indices)):
        if silent_indices[i] - silent_indices[i - 1] == 1:
            continue
        end = silent_indices[i - 1]
        duration = (end - start) / sr
        silent_durations.append((start, end, duration))
        start = silent_indices[i]
    # 处理最后一个静音段
    end = silent_indices[-1]
    duration = (end - start) / sr
    silent_durations.append((start, end, duration))

    # 只考虑首尾的静音段
    silent_to_cut = []
    if silent_durations:
        first_start, first_end, first_duration = silent_durations[0]
        if first_duration > silence_duration_threshold:
            silent_to_cut.append((first_start, first_end))
        last_start, last_end, last_duration = silent_durations[-1]
        if last_duration > silence_duration_threshold:
            silent_to_cut.append((last_start, last_end))

    # 找到非静音段的起始和结束索引
    non_silent_segments = []
    start = 0
    if silent_to_cut:
        if silent_to_cut[0][0] == silent_durations[0][0]:  # 处理开始处的静音段
            start = silent_to_cut[0][1] + 1
        if silent_to_cut[-1][0] == silent_durations[-1][0]:  # 处理结束处的静音段
            end = silent_to_cut[-1][0]
            non_silent_segments.append((start, end))
        else:
            non_silent_segments.append(start, len(waveform[0]))
    else:  # 如果没有需要剪切的静音段，直接复制整个音频
        non_silent_segments = [(0, len(waveform[0]))]

    # 拼接非静音段
    new_waveform = torch.cat([waveform[:, start:end] for start, end in non_silent_segments], dim=1)
    # 保存新的音频文件
    torchaudio.save(output_path, new_waveform, sr)


gpu_id = 7
def init_model_my():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    config_path = "/home/node54_tmpdata/xlgeng/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/update_data/epoch_1_with_token/epoch_11.yaml"
    config_path = "/home/work_nfs15/asr_data/ckpt/understanding_model/step_24999.yaml"

    checkpoint_path = "/home/node54_tmpdata/xlgeng/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/update_data/epoch_1_with_token/epoch_11.pt"
    checkpoint_path = "/home/work_nfs15/asr_data/ckpt/understanding_model/step_24999.pt"
    args = GxlNode({
        "checkpoint": checkpoint_path,
    })
    configs = utils_file.load_dict_from_yaml(config_path)
    model, configs = init_model(args, configs)
    model = model.cuda(gpu_id)
    tokenizer = init_tokenizer(configs)
    print(model)
    return model, tokenizer


model, tokenizer = init_model_my()

def do_resample(input_wav_path, output_wav_path):
    """"""
    print(f'input_wav_path: {input_wav_path}, output_wav_path: {output_wav_path}')
    waveform, sample_rate = torchaudio.load(input_wav_path)
    # 检查音频的维度
    num_channels = waveform.shape[0]
    # 如果音频是多通道的，则进行通道平均
    if num_channels > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = torchaudio.transforms.Resample(
        orig_freq=sample_rate, new_freq=16000)(waveform)
    utils_file.makedir_for_file(output_wav_path)
    torchaudio.save(output_wav_path, waveform, 16000)


# input_wav_path = "/home/work_nfs15/asr_data/data/asr_test_sets/speechio_15/wav/3pwxGLuHyC8_0189-001.wav"
# input_prompt = "将这段音频的语音内容详细记录为文字稿。"
def do_decode(input_wav_path, input_prompt):
    timestamp_ms = int(time.time() * 1000)
    now_file_tmp_path_resample = f'/home/xlgeng/.cache/.temp/{timestamp_ms}_resample.wav'
    do_resample(input_wav_path, now_file_tmp_path_resample)
    tmp_vad_path = f'/home/xlgeng/.cache/.temp/{timestamp_ms}_vad.wav'
    remove_silence_torchaudio_ends(now_file_tmp_path_resample, tmp_vad_path)
    input_wav_path  = tmp_vad_path
    waveform, sample_rate = torchaudio.load(input_wav_path)
    waveform = waveform.squeeze(0)  # (channel=1, sample) -> (sample,)
    print(f'wavform shape: {waveform.shape}, sample_rate: {sample_rate}')
    window = torch.hann_window(400)
    stft = torch.stft(waveform,
                      400,
                      160,
                      window=window,
                      return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = torch.from_numpy(
        librosa.filters.mel(sr=sample_rate,
                            n_fft=400,
                            n_mels=80))
    mel_spec = filters @ magnitudes

    # NOTE(xcsong): https://github.com/openai/whisper/discussions/269
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    feat = log_spec.transpose(0, 1)
    feat_lens = torch.tensor([feat.shape[0]], dtype=torch.int64).to(gpu_id)
    feat = feat.unsqueeze(0).to(gpu_id)
    # feat = feat.half()
    # feat_lens = feat_lens.half()
    res_text = model.generate(wavs=feat, wavs_len=feat_lens, prompt=input_prompt)[0]
    print("耿雪龙哈哈：", res_text)
    print(f"wav_path: {input_wav_path}, prompt:{input_prompt}")
    return res_text


# if __name__ == "__main__":
#     input_wav_path = "/tmp/gradio/e4df3769509c79fbac84161611f2de61e65726acc71c511d01b5944141cf9611/audio.wav"
#     output_wav_path = utils_file.do_get_fake_file() +'.wav'
#     do_resample(input_wav_path, output_wav_path)


# 创建Gradio界面
iface = gr.Interface(
    fn=do_decode,  # 调用的推理函数
    inputs=[
        gr.Audio(label="录音", type="filepath"),
        gr.Textbox(label="Prompt"),  # 输入框：文本

    ],
    outputs="text"  # 输出：文本（返回“哈哈”）
)

# 启动Gradio界面并共享
iface.launch()  # 设置 share=True，会生成一个公网可访问的链接
