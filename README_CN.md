![image](https://github.com/user-attachments/assets/7dbd0fb0-d7e9-46aa-8c81-e97780ffdbff)<p align="left">
        <a >中文</a> &nbsp｜ &nbsp <a href="README.md">English</a>&nbsp&nbsp
</p>
<p align="center">
   <h1>OSUM: Advancing Open Speech Understanding Models with Limited Resources in Academia</h1>
</p>

Xuelong Geng, Kun Wei, Qijie Shao, Shuiyun Liu*, Zhennan Lin*, Zhixian Zhao*, Guojian Li*, Wenjie Tian*, Peikun Chen, Yangze Li, Pengcheng Guo, Mingchen Shao, Shuiyuan Wang, Yuang Cao, Chengyou Wang, Tianyi Xu, Yuhang Dai, Xinfa Zhu, Yue Li, Li Zhang, Lei Xie†


<p align="center">
    <img src="images/SUM.png" width="400"/>
<p>

<p align="center">
OSUM <a href=""> 敬请期待</a> </a>&nbsp
<br>
📑 <a href="">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://aslp-lab.github.io/OSUM.github.io/">Demo</a> &nbsp&nbsp | &nbsp&nbsp 💬 <a href="">WeChat (微信)</a>&nbsp&nbsp 
</p>


大型语言模型（LLMs）在各种下游任务上取得了显著进展，激发了语音理解语言模型（SULMs）的发展，以实现全面的基于语音的交互。然而，目前大多数先进的 SULMs 由工业界开发，依赖于大规模数据集和计算资源，这些资源对学术界而言并不易得。此外，训练细节的缺乏透明性也进一步阻碍了创新。在本研究中，我们提出了 OSUM，一种开放的语音理解模型，旨在探索在受限的学术资源下训练 SULMs 的潜力。OSUM 模型将 Whisper 编码器与 Qwen2 大语言模型相结合，支持多种语音任务，包括自动语音识别（ASR）、带时间戳的语音识别（SRWT）、语音事件检测（VED）、语音情感识别（SER）、说话风格识别（SSR）、说话人性别分类（SGC）、说话人年龄预测（SAP）以及语音到文本聊天（STTC）。通过采用 ASR+X 训练策略，OSUM 通过同时优化 ASR 与目标任务，实现了高效且稳定的多任务训练。除了提供强大的性能外，OSUM 还强调透明性，公开提供数据准备和训练方法，为学术界提供宝贵的见解和实践指导。通过这一举措，我们旨在加速先进 SULM 技术的研究与创新。

## Architecture

OSUM的架构与任务概述。

<p align="center">
    <img src="images/system.png" width="90%"/>
<p>

## News and Updates
* checkpoint和测试页面即将发布，敬请期待


<br>

## Evaluation
Qwen2-Audio 和我们的 OSUM 模型比较。在大多数任务中，尽管 OSUM 使用的计算资源和训练数据明显更少，但它的表现优于 Qwen2-Audio。
<p align="center">
    <img src="images/radar.png" width="80%"/>
<p>

在公共和内部测试集上的 ASR 任务评估结果中，加粗字体表示同一测试集中的最佳结果。所有内部测试结果均由我们自行推理得出。


<p align="center">
    <img src="images/res_asr.png" width="90%"/>
<p>

在公共和内部测试集上的多任务评估结果中，每个测试集的最佳结果均以加粗字体突出显示。以蓝色字体显示的结果以及内部测试集上的结果，均使用我们自行推理的原始发布模型得出。
<p align="center">
    <img src="images/res_multi.png" width="90%"/>
<p>


<!--  We have provided **all** evaluation scripts to reproduce our results. Please refer to [eval_audio/EVALUATION.md](eval_audio/EVALUATION.md) for details.
  --> 


## Requirements
```
pip install requirements_little.txt
```

<!--## Quickstart
## Demo
### Web UI

## Citation
-->
## Contact Us

如果您有兴趣向我们的研究团队或产品团队留言，欢迎发送电子邮件至 `xlgeng@mail.nwpu.edu.cn`。
<p align="center">
    <img src="images/ASLP.jpg" width="400"/>
<p>
