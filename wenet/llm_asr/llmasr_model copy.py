import logging
import os

import torchaudio
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from wenet.transformer.encoder import TransformerEncoder
from wenet.llm_asr.utils4llmasr import *
from gxl_ai_utils.utils import utils_file

from wenet.llm_asr.downsampler import get_downsampler, LyzConv1dSubsampling
from wenet.utils.mask import make_pad_mask
# import torch_npu
# from torch_npu.contrib import transfer_to_npu

from msprobe.pytorch import seed_all,PrecisionDebugger

class LLMASR_Model(nn.Module):
    def __init__(self,
                 encoder,
                 encoder_output_dim,
                 llm_path,
                 lora=True, lora_alpha=32,lora_rank=8, lora_dropout=0.1,
                 prompt_pattern="{}：<Speech><SpeechHere></Speech>",
                 # "USER: <Speech><SpeechHere></Speech> {}\nASSISTANT:"
                 is_inference=False,
                 downsample_rate=1,
                 llm_embed_dim=4096,
                 task_num=2,
                 adapter_type='lyz',
                 speech_token_num=0,
                 train_speech_out=False):
        """"""
        super().__init__()
        self.downsample_rate = downsample_rate

        self.encoder = encoder
        self.ln_speech = nn.LayerNorm(encoder_output_dim)

        # 连接层, 51.6M
        if adapter_type == 'gxl':
            self.speech_transformer = TransformerEncoder(
                input_size=encoder_output_dim,
                output_size=encoder_output_dim,
                attention_heads=4,
                linear_units=2560,
                num_blocks=4,
                dropout_rate=0.1,
                positional_dropout_rate=0.1,
                attention_dropout_rate=0.0,
                input_layer="linear",
                pos_enc_layer_type="abs_pos",
                normalize_before=True
            )
        else:
            self.speech_transformer = None

        # LLM,
        self.low_resource = False
        if not self.low_resource:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llm_path,
                # torch_dtype=torch.float32 if is_inference else torch.float16,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                output_hidden_states=True,
            )
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llm_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True,
                output_hidden_states=True,
            )


        self.max_length = 400
        self.min_length = 1
        self.num_beams = 4
        self.do_sample = True
        self.top_p = 0.0
        self.top_k = 0
        self.repetition_penalty = 1.05
        self.length_penalty = 1.0
        self.temperature = 1.0
        self.IGNORE_ID = -100

        # lora
        self.lora = lora
        if lora:
            utils_file.logging_limit_print("耿雪龙： 使用lora了")
            target_modules = ['W_pack', 'o_proj', 'gate_proj', 'down_proj']
            if is_inference:
                self.peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=True,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                )
            else:
                self.peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path, use_fast=False, trust_remote_code=True)
        """
        设置分词器的pad_token和padding的方向。
        """
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.padding_side = "right"
        

        if hasattr(self.llama_model.config, 'hidden_size'):
            utils_file.logging_limit_print(f"self.llama_model.config.hidden_size: {self.llama_model.config.hidden_size}")
            if adapter_type == 'lyz':
                self.down_sample_2 = LyzConv1dSubsampling(encoder_output_dim, self.llama_model.config.hidden_size)
            elif adapter_type == 'gxl':
                self.down_sample_2 = get_downsampler(downsample_rate, encoder_output_dim)
                self.speech_llama_proj = nn.Linear(
                    encoder_output_dim, self.llama_model.config.hidden_size)
            # self.task_embeddings = torch.nn.Embedding(task_num, self.llama_model.config.hidden_size)
        else:
            raise NotImplementedError("self.llama_model.config.hidden_size not exist")
        
        self.embed_tokens = self.llama_model.model.model.embed_tokens if self.lora else self.llama_model.model.embed_tokens
        self.lm_head = self.llama_model.model.lm_head if self.lora else self.llama_model.lm_head

        self.speech_token_num = speech_token_num
        # init speech token module
        if speech_token_num > 0:
            utils_file.logging_info(f'耿雪龙： 进行语音token生成任务， speech_token_num: {speech_token_num}')
            self.speech_token_emded = torch.nn.Embedding(speech_token_num+2, self.llama_model.config.hidden_size)
            self.speaker_head = torch.nn.Linear(self.llama_model.config.hidden_size, speech_token_num)
        else:
            # 不做任何处理
            self.speaker_head =nn.Identity()
            self.speech_token_emded = nn.Identity()
        self.train_speech_out = train_speech_out
        utils_file.logging_info(f'耿雪龙： 是否进行语音输出训练：{self.train_speech_out}')
        self.loss_fct = CrossEntropyLoss(reduction='mean')
        # self.debugger = PrecisionDebugger(config_path='./do_align_test/config_gpu.json', model=self.encoder)


    def forward(self,
                batch,
                device,
                ):
        """"""
        rank = int(os.environ.get('RANK', 0))
        labels = batch['target'].to(device)
        labels_lengths = batch['target_lengths'].to(device)
        wavs = batch['feats'].to(device)
        # if rank == 0:
        #     utils_file.logging_limit_print(
        #         f'wavs shape: {wavs.shape},第一帧的前20个数字：\n{wavs[0][0][:20]}')

        wavs_len = batch['feats_lengths'].to(device)
        output_type = batch['output_type']
        assert output_type in ['text','speech_token'], f"output_type:{output_type} not support"
        # lang speaker emotion gender -> List<str>
        # duration -> List<float>
        # 如有用到该数据,需要使用对应的str_to_id进行映射
        if 'lang' in batch:
            lang = batch['lang']
        else:
            lang = None
        if 'speaker' in batch:
            speaker = batch['speaker']
        else:
            speaker = None
        if 'emotion' in batch:
            emotion = batch['emotion']
        else:
            emotion = None
        if 'gender' in batch:
            gender = batch['gender']
        else:
            gender = None
        if 'duration' in batch:
            duration = batch['duration']
        else:
            duration = None
        if 'task' in batch:
            task = batch['task']
        else:
            task = None

        # utils_file.logging_limit_print('进入 llmasr forward() ,首先来看一下输入')
        # utils_file.logging_limit_print('wavs.shape:', wavs.shape)
        # utils_file.logging_limit_print('wavs_len.shape:', wavs_len.shape)
        # utils_file.logging_limit_print('wavs_len:', wavs_len)
        # utils_file.logging_limit_print('labels.shape:', labels.shape)
        # utils_file.logging_limit_print('labels_lengths.shape:', labels_lengths.shape)
        # utils_file.logging_limit_print('output_type:', output_type)
        # utils_file.logging_limit_print('观看结束')
        
        # speech inputs
        speech_embeds, speech_masks = self.get_embedding_from_wav(wavs, wavs_len)
        B = speech_embeds.size(0)
        speech_target = torch.full(speech_masks.shape, self.IGNORE_ID).to(
                                                            speech_embeds.device)
        # add bos and eos
        speech_embeds, speech_masks, speech_target = self._add_bos_eos(0+self.speech_token_num, 1+self.speech_token_num, 
                                    speech_embeds, speech_masks, speech_target)

        # template mode
        # if batch.get('role', None) is not None:
        #         chat_prefix = extra_inputs['role']
        #         assert chat_prefix.size(0) == B
        # else:
        #     chat_prefix = self.chat_template['role'].repeat(B, 1).to(
        #                                             speech_embeds.device)
        # chat_prefix = self.chat_template['prefix'].repeat(B, 1).to(
        #                                         speech_embeds.device)
        # chat_suffix = self.chat_template['suffix'].repeat(B, 1).to(
        #                                         speech_embeds.device)
        # chat_prefix_embeds = self.llm_decoder.transformer.wte(chat_prefix)
        # chat_suffix_embeds = self.llm_decoder.transformer.wte(chat_suffix)
        # chat_prefix_mask = torch.full(chat_prefix.shape, 
        #                     True).to(speech_embeds.device)
        # chat_prefix_target = torch.full(chat_prefix.shape, 
        #                     self.IGNORE_ID).to(speech_embeds.device)
        # chat_suffix_mask = torch.full(chat_suffix.shape, 
        #                     True).to(speech_embeds.device)
        # chat_suffix_target = torch.full(chat_suffix.shape, 
        #                     self.IGNORE_ID).to(speech_embeds.device)

        # prompt
        if 'prompt' in batch:
            prompt = batch['prompt'].to(device)
            prompt_lengths = batch['prompt_lengths'].to(device)
            prompt_pad_mask = make_pad_mask(prompt_lengths) # B, L
            prompt = prompt.masked_fill(prompt_pad_mask, self.tokenizer.eos_token_id)
            prompt_embeds = self.embed_tokens(prompt) # B, L, D
            prompt_target = torch.full(prompt.shape, self.IGNORE_ID).to(
                                                        speech_embeds.device) # B, L
            prompt_mask = ~prompt_pad_mask
            # chat_prefix_embeds = torch.cat([chat_prefix_embeds, prompt_embeds], dim=1)
            # chat_prefix_mask = torch.cat([chat_prefix_mask, prompt_mask], dim=1)
            # chat_prefix_target = torch.cat([chat_prefix_target, prompt_target], dim=1)

        # label
        labels_pad_mask = make_pad_mask(labels_lengths) # B, L
        labels = labels.masked_fill(labels_pad_mask, 0)
        if output_type == 'speech_token':
            labels_embeds = self.speech_token_emded(labels)
        else:
            labels_embeds = self.embed_tokens(labels) # B, L, D
        if output_type == 'speech_token':
            utils_file.logging_limit_print(f'进行label修改，修改前label', labels.shape, labels[0][-1], labels[0][0])
            labels = labels + 152064
            utils_file.logging_limit_print(f'进行label修改，修改后label', labels.shape, labels[0][-1], labels[0][0])
        labels_target = labels.masked_fill(labels_pad_mask, self.IGNORE_ID) # B, L
        labels_mask = ~labels_pad_mask


        # concat
        inputs_embeds = torch.cat([prompt_embeds, speech_embeds, 
                                       labels_embeds], dim=1)
        attention_mask = torch.cat([prompt_mask, speech_masks, 
                                        labels_mask], dim=1)
        target = torch.cat([prompt_target, speech_target, 
                                        labels_target], dim=1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            # labels=target,
            attention_mask=attention_mask,
            position_ids=position_ids.to(inputs_embeds.device)
        )
        # utils_file.logging_limit_print(f'耿雪龙 output_type: {output_type}')
        hidden_states = outputs['hidden_states'][-1]
        # utils_file.logging_limit_print(f'hidden_states: {hidden_states.shape}')
        # if rank == 0:
        #     utils_file.logging_limit_print(
        #         f'out of llm shape: {hidden_states.shape},第一帧的前20个数字：\n{hidden_states[0][0][:20]}')

        logits = self.lm_head(hidden_states)
        # if rank == 0:
        #     utils_file.logging_limit_print(
        #         f'out of lm_head shape: {logits.shape},第一帧的前20个数字：\n{logits[0][0][:20]}')

        logits2 = self.speaker_head(hidden_states)
        # if rank == 0:
        #     utils_file.logging_limit_print(
        #         f'out of speaker_head shape: {logits2.shape},第一帧的前20个数字：\n{logits2[0][0][:20]}')


        # 在最后一维进行拼接（cat）操作
        combined_logits = torch.cat([logits, logits2], dim=-1)

        shift_logits = combined_logits[..., :-1, :].contiguous()
        # utils_file.logging_limit_print(f'shift_logits.shape', shift_logits.shape)
        shift_target = target[..., 1:].contiguous()
        # utils_file.logging_limit_print(f'shift_target', shift_target)
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, combined_logits.shape[-1])  # 注意这里维度的调整，根据logits2的维度相应改变
        shift_target = shift_target.view(-1)
        # Enable model parallelism
        shift_target = shift_target.to(shift_logits.device)
        loss = self.loss_fct(shift_logits, shift_target)
        loss.requires_grad_(True)
        return {"loss": loss}


    def generate(
            self,
            wavs,
            wavs_len,
            prompt,
    ):
        speech_embeds, speech_masks = self.get_embedding_from_wav(wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(0+self.speech_token_num, 1+self.speech_token_num, 
                                    speech_embeds, speech_masks, None)
        prompt = self.tokenizer([prompt], return_tensors="pt"
                                    )['input_ids'].to(speech_embeds.device)
        prompt_embeds = self.embed_tokens(prompt)

        embeds = torch.cat([prompt_embeds, speech_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)
        
        if self.embed_tokens.weight.dtype == torch.float16:
            utils_file.logging_limit_print('generate(): self.embed_tokens.weight.dtype == torch.float16')
            embeds = embeds.to(torch.float16)
            atts = atts.half()
        outputs = self.llama_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=self.max_length,
            num_beams=self.num_beams,
            do_sample=self.do_sample,
            min_length=self.min_length,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            temperature=self.temperature,
            attention_mask=atts,
            eos_token_id=151643,
            pad_token_id=-100,
        )

        output_text = self.tokenizer.batch_decode(outputs, add_special_tokens=False, skip_special_tokens=True)

        return output_text

    def get_embedding_from_wav(self, wavs, wavs_len):
        """"""
        # utils_file.logging_limit_print('get_embedding_from_wav(): wavs.shape:', wavs.shape)
        # utils_file.logging_limit_print('get_embedding_from_wav(): wavs_len.shape:', wavs_len.shape)
        rank = int(os.environ.get('RANK', 0))
        # self.debugger.start()
        encoder_out, encoder_mask = self.encoder(wavs, wavs_len)
        # self.debugger.stop()
        # self.debugger.step()
        # if rank == 0:
        #     utils_file.logging_limit_print(f'encoder out shape: {encoder_out.shape},encoder的第一帧的前20个数字：\n{encoder_out[0][0][:20]}')

        # utils_file.logging_limit_print(
        #     'get_embedding_from_wav(): speech_embeds.shape,by  self.encoder(wavs, wavs_len):',
        #     encoder_out.shape)

        speech_embeds, encoder_mask = self.down_sample_2(encoder_out, encoder_mask)
        # if rank == 0:
        #     utils_file.logging_limit_print(f'out of down_sample_2 shape: {speech_embeds.shape},encoder的第一帧的前20个数字：\n{speech_embeds[0][0][:20]}')


        # utils_file.logging_limit_print(
        #     'get_embedding_from_wav(): speech_embeds.shape,by  self.down_sample_2(speech_embeds):', speech_embeds.shape)
        # # max_utt_len = speech_embeds.size(1)
        # filled_wavs_len = torch.ones(speech_embeds.size(0)) * max_utt_len
        # filled_wavs_len = filled_wavs_len.to(speech_embeds.device)
        if self.speech_transformer is not None:
            filled_wavs_len = encoder_mask.squeeze(1).sum(-1)
            speech_embeds, encoder_mask = self.speech_transformer(speech_embeds, filled_wavs_len)
            # if rank == 0:
            #     utils_file.logging_limit_print(
            #         f'out of link shape: {speech_embeds.shape},encoder的第一帧的前20个数字：\n {speech_embeds[0][0][:20]}')

            # utils_file.logging_limit_print(
        #     'get_embedding_from_wav(): speech_embeds.shape,by  self.speech_transformer(speech_embeds, speech_lens):',
        #     speech_embeds.shape)
            speech_embeds = self.speech_llama_proj(speech_embeds)
            # if rank == 0:
            #     utils_file.logging_limit_print(
            #         f'out of speech_llama_proj shape: {speech_embeds.shape},encoder的第一帧的前20个数字：\n {speech_embeds[0][0][:20]}')

        # utils_file.logging_limit_print(
        #     'get_embedding_from_wav(): speech_embeds.shape,by  self.speech_llama_proj(speech_embeds):',
        #     speech_embeds.shape)
        
        return speech_embeds, encoder_mask.squeeze(1)

    def get_embedding_from_text(self, text):
        text_id = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        ).to(
            self.embed_tokens.weight.device).input_ids
        text_embeds = self.embed_tokens(text_id)
        return text_embeds

    def get_embeds_from_wav_path(self, wav_path):
        wav_i2_path = wav_path
        utils_file.logging_limit_print('get_embeds_from_wav_path(): wav_i2_path:', wav_i2_path)
        waveform_i2, _ = torchaudio.load(wav_i2_path)
        utils_file.logging_limit_print('get_embeds_from_wav_path(): waveform_i2.shape:', waveform_i2.shape)
        if len(waveform_i2.shape) != 1:
            waveform_i2 = waveform_i2[0]
        waveform_i2 = waveform_i2.to(self.embed_tokens.weight.device)
        wavs_len_i2 = torch.tensor([len(waveform_i2)], device=self.embed_tokens.weight.device, dtype=torch.int32)
        wavs_i2 = waveform_i2.unsqueeze(0)
        sample_i2_embeds = self.get_embedding_from_wav(wavs_i2, wavs_len_i2)
        utils_file.logging_limit_print('get_embeds_from_wav_path(): sample_i2_embeds.shape:', sample_i2_embeds.shape)
        return sample_i2_embeds

    def _add_bos_eos(self, bos, eos, inputs_embeds, attention_mask, target=None):
        B = len(inputs_embeds)
        bos_eos_target = torch.full([B, 1], self.IGNORE_ID).to(inputs_embeds.device) # B,1
        bos_eos_mask = torch.full([B, 1], True).to(inputs_embeds.device) # B, 1

        if bos is not None:
            bos_embed = self.speech_token_emded(torch.full([B, 1], 
                                            bos).to(inputs_embeds.device)) # B, 1, D
            inputs_embeds = torch.cat((bos_embed, inputs_embeds), 1) # B, (1+T), D
            attention_mask = torch.cat((bos_eos_mask, attention_mask), 1) # B, (1+T)
            if target is not None:
                target = torch.cat((bos_eos_target, target), 1) # B, (1+T), D

        if eos is not None:
            eos_embed = self.speech_token_emded(torch.full([B, 1], 
                                            eos).to(inputs_embeds.device)) # B, 1, D
            inputs_embeds = torch.cat((inputs_embeds, eos_embed), 1) # B, (1+T+1), D   
            attention_mask = torch.cat((attention_mask, bos_eos_mask), 1) # B, (1+T+1)
            if target is not None:
                target = torch.cat((target, bos_eos_target), 1) # B, (1+T+1), D
        
        return inputs_embeds, attention_mask, target
