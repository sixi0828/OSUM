import sys

import torch
import torchaudio
from torch import dtype

sys.path.insert(0,'../../../../')
from gxl_ai_utils.utils import utils_file
from wenet.utils.init_tokenizer import init_tokenizer
from gxl_ai_utils.config.gxl_config import GxlNode
from wenet.utils.init_model import init_model
import logging
import librosa
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
config_path = "/home/node54_tmpdata/xlgeng/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/update_data/epoch_1_with_token/epoch_11.yaml"
checkpoint_path = "/home/node54_tmpdata/xlgeng/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/update_data/epoch_1_with_token/epoch_11.pt"
args = GxlNode({
    "checkpoint": checkpoint_path,
})
configs = utils_file.load_dict_from_yaml(config_path)
model, configs = init_model(args, configs)
gpu_id = 3
model = model.cuda(gpu_id)
tokenizer = init_tokenizer(configs)
print(model)
input_wav_path = "/home/work_nfs15/asr_data/data/asr_test_sets/speechio_15/wav/3pwxGLuHyC8_0189-001.wav"
input_prompt = "将这段音频的语音内容详细记录为文字稿。"
def do_decode(input_wav_path, input_prompt):
    waveform, sample_rate = torchaudio.load(input_wav_path)
    waveform = waveform.squeeze(0)  # (channel=1, sample) -> (sample,)
    # print(f'wavform shape: {waveform.shape}')
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
    return res_text

if __name__ == "__main__":
    res_text = do_decode(input_wav_path, input_prompt)
    print(res_text)

