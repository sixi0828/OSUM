 <p align="left">
        <a href="README_CN.md">中文</a> &nbsp｜ &nbsp English&nbsp&nbsp
</p>
<br><br>

<p align="center">
    <img src="images/ASLP.jpg" width="400"/>
<p>

<p align="center">
OSUM <a href=""> 敬请期待</a> </a>&nbsp
<br>
📑 <a href="">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://aslp-lab.github.io/OSUM.github.io/">Demo</a> &nbsp&nbsp | &nbsp&nbsp 💬 <a href="images/wechat.png">WeChat (微信)</a>&nbsp&nbsp 
</p>


We present OSUM, an open-source SULM designed to encourage broader research participation with minimal resource requirements. OSUM integrates a Whisper speech encoder, fine-tuned on a multi-task dataset, with a Qwen2 LLM. It supports various speech tasks, including <b>ASR</b>, <b>SRWT</b>, <b>VED</b>, <b>SER</b>, <b>SSR</b>, <b>SGC</b>, <b>SAP</b>, and <b>STTC</b>. Notably, <b>SSR</b> enhances speech generation by ensuring appropriate styles for natural interactions.  

To improve training efficiency and stability, we adopt an ASR+X strategy, where an auxiliary ASR task is trained alongside the primary task (X), accelerating modality alignment between text and audio. OSUM uses 44,100 hours of training data and achieves competitive performance. It is trained on Nvidia A6000 GPUs and Huawei Ascend NPUs, supporting inference on both platforms.  

Our goal is to foster transparency and accelerate SULM research by providing accessible tools and resources to the community.

## Architecture

The overview of the architecture and tasks of OSUM.

<p align="center">
    <img src="images/system.png" width="80%"/>
<p>

## News and Updates
* Checkpoint and test page coming soon. Stay tuned!


<br>

## Evaluation
 Comparison of Qwen2-Audio and our OSUM model. In most tasks, OSUM achieves a better
performance than Qwen2-Audio despite using significantly fewer computational resources and training data.
<p align="center">
    <img src="images/radar.png" width="80%"/>
<p>

Evaluation results of ASR tasks on public and internal test sets. The bold font represents the best
result among the same test set. All internal results are inferred by ourselves.
<p align="center">
    <img src="images/res_asr.png" width="80%"/>
<p>


Evaluation results of multi-tasking on public and internal test sets. The best results for each test set
are highlighted in bold font. Results shown in blue font, as well as those on internal test sets, are inferred
using the original released model by ourselves.
<p align="center">
    <img src="images/res_multi.png" width="80%"/>
<p>

<!--  We have provided **all** evaluation scripts to reproduce our results. Please refer to [eval_audio/EVALUATION.md](eval_audio/EVALUATION.md) for details.
  --> 


## Requirements
```
pip install requirements_little.txt
```
<!-- 
## Quickstart
## Demo
### Web UI

## Citation
 -->
## Contact Us

If you are interested to leave a message to either our research team or product team, feel free to send an email to `g3349495429@163.com`.
