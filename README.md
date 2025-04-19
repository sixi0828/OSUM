 <p align="left">
        <a href="README_CN.md">中文</a> &nbsp｜ &nbsp English&nbsp&nbsp ｜ <a href="README_JP.md">日本語</a> &nbsp
</p>
<p align="center">
   <h1>OSUM: Advancing Open Speech Understanding Models with Limited Resources in Academia</h1>
</p>

Xuelong Geng, Kun Wei, Qijie Shao, Shuiyun Liu*, Zhennan Lin*, Zhixian Zhao*, Guojian Li*, Wenjie Tian*, Peikun Chen, Yangze Li, Pengcheng Guo, Mingchen Shao, Shuiyuan Wang, Yuang Cao, Chengyou Wang, Tianyi Xu, Yuhang Dai, Xinfa Zhu, Yue Li, Li Zhang, Lei Xie†




<p align="center">
    <img src="images/SUM.png" width="400"/>
<p>


<p align="center">
 <a href="https://huggingface.co/spaces/ASLP-lab/OSUM"> Huggingface Test Page</a> </a>&nbsp
<br>
📑 <a href="https://arxiv.org/abs/2501.13306v2">Paper (v2.0)</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://aslp-lab.github.io/OSUM.github.io/">Demo</a> &nbsp&nbsp | &nbsp&nbsp 💬 <a href="images/introduction.md">WeChat (微信)</a>&nbsp&nbsp 
</p>

 <h3 align="center">OSUM is pronounced as ‘awesome’ (/ˈɔː.səm/).</h3>


Large Language Models (LLMs) have made significant progress in various downstream tasks, inspiring the development of Speech Understanding Language Models (SULMs) to enable comprehensive speech-based interactions. However, most advanced SULMs are developed by the industry, leveraging large-scale datasets and computational resources that are not readily available to the academic community. Moreover, the lack of transparency in training details creates additional barriers to further innovation. In this study, we present OSUM, an Open Speech Understanding Model designed to explore the potential of training SLUMs under constrained academic resources. The OSUM model combines a Whisper encoder with a Qwen2 LLM and supports a wide range of speech tasks, including speech recognition (ASR), speech recognition with timestamps (SRWT), vocal event detection (VED), speech emotion recognition (SER), speaking style recognition (SSR), speaker gender classification (SGC), speaker age prediction (SAP), and speech-to-text chat (STTC). By employing an ASR+X training strategy, OSUM achieves efficient and stable multi-task training by simultaneously optimizing ASR alongside target tasks.
Beyond delivering strong performance, OSUM emphasizes transparency by providing openly available data preparation and training methodologies, offering valuable insights and practical guidance for the academic community. By doing so, we aim to accelerate research and innovation in advanced SULM technologies.

## Architecture

The overview of the architecture and tasks of OSUM.

<p align="center">
    <img src="images/system.png" width="80%"/>
<p>

## News and Updates

### 2025.2.16 🎉 We updated the technical report [OSUM technical report v2.0](https://arxiv.org/abs/2501.13306v2) and released the [checkpoint](https://huggingface.co/ASLP-lab/OSUM), and the online [test page](https://huggingface.co/spaces/ASLP-lab/OSUM) on hugging face.
In technical report v2.0, the OSUM model has gone through more training steps and the training data volume has increased to 50.5K hours (as compared to 44.1K hours in v1.0) 
- 3000 hours of speech gender classification (SGC) data, which includes 1500 hours of existing data augmented with noise, and another 1500 hours of new data.
- Speaker age prediction (SAP) data expansion: The original 3400 hours of age prediction data were augmented with noise, doubling the volume to 6800 hours.
### 2025.1.22 🔥 We released the [OSUM technical report v1.0](https://arxiv.org/abs/2501.13306v1).

<br>

## Evaluation
 Comparison of Qwen2-Audio and our OSUM model. In most tasks, OSUM achieves a better
performance than Qwen2-Audio despite using significantly fewer computational resources and training data.
<p align="center">
    <img src="images/radar.jpg" width="80%"/>
<p>

Evaluation results of ASR tasks on public and internal test sets. The bold font represents the best
result among the same test set. All internal results are inferred by ourselves.
<p align="center">
    <img src="images/res_asr.jpg" width="80%"/>
<p>


Evaluation results of multi-tasking on public and internal test sets. The best results for each test set
are highlighted in bold font. Results shown in blue font, as well as those on internal test sets, are inferred
using the original released model by ourselves.
<p align="center">
    <img src="images/res_multi.jpg" width="80%"/>
<p>

<!--  We have provided **all** evaluation scripts to reproduce our results. Please refer to [eval_audio/EVALUATION.md](eval_audio/EVALUATION.md) for details.
  --> 


## Requirements
```
pip install requirements.txt
```
<!-- 
## Quickstart
## Demo
### Web UI
-->
 <p align="left">
       How to use the OSUM framework for inference and training? Please refer to <a href="examples/osum/README.md">here</a>
</p>

## License Agreement

We use the Apache 2.0 license. Researchers and developers are free to use the codes and model weights of our OSUM, even for commercial use. Check the license at [LICENSE.txt](LICENSE.txt) for more details.
<br>

## Citation
```
@article{geng2025osum,
  title={{OSUM}: {Advancing} Open Speech Understanding Models with Limited Resources in Academia},
  author={Geng, Xuelong and Wei, Kun and Shao, Qijie and Liu, Shuiyun and Lin, Zhennan and Zhao, Zhixian and Li, Guojian and Tian, Wenjie and Chen, Peikun and Li, Yangze and others},
  journal={arXiv preprint arXiv:2501.13306},
  year={2025}
}
```
## Contact Us

If you are interested in leaving a message to our research team, feel free to email `xlgeng@mail.nwpu.edu.cn`.
<p align="center">
    <a href="http://www.nwpu-aslp.org/">
        <img src="images/ASLP.jpg" width="400"/>
    </a>
</p>
<p align="center">
    <a href="https://wenet.org.cn/">
        <img src="images/wenet.png" width="400"/>
    </a>
</p>
