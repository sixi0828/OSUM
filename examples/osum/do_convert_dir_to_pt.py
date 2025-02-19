import torch
import torch_npu
exp_dir = "/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/epoch_10_11_with_speech_gxl_with_asr-chat"
exp_dir = "/mnt/sfs/asr/ckpt/qwen2-7B-instruct_multi_task_4_gxl_adapter/epoch_0"
exp_dir = "/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/epoch_12_13_with_speech_gxl_with_asr-chat"
exp_dir = "/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/only_emotion_from_epoch11_with_ssl_vec"
exp_dir = "/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/epoch_13_with_speech_gxl_with_asr-chat_full-data"
exp_dir = "/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/only_emotion_from_epoch11"
exp_dir = "/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/epoch_13_with_asr-chat_full_data"
exp_dir = "/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/epoch_14_with_asr-chat_full_data"
exp_dir = "/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/epoch_14_with_asr-chat_full_data_50percent_pureX"
pt_name = "step_4999"
weight_dict = torch.load(f"{exp_dir}/{pt_name}/mp_rank_00_model_states.pt",map_location=torch.device('cpu'))['module']
print(weight_dict.keys())
torch.save(weight_dict, f"{exp_dir}/{pt_name}.pt")
