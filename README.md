
<h1 align="center">RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics</h1>

<h4 align="center">From words to exactly where you mean — with RoboRefer</h4>


<p align="center">
  <a href="https://arxiv.org/abs/2403.12037">
    <img src="https://img.shields.io/badge/arXiv-TODO-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://zhoues.github.io/RoboRefer/">
    <img src="https://img.shields.io/badge/%F0%9F%8F%A0%20Project-Homepage-blue" alt="Project Homepage">
  </a>
  <a href="https://huggingface.co/datasets/BAAI/RefSpatial-Bench">
    <img src="https://img.shields.io/badge/🤗%20Benchmark-RefSpatial--Bench-green.svg" alt="Benchmark">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-Stay%20tuned-yellow" alt="Weights">
  </a>
</p>


<div style="text-align: center; background-color: white;">
    <img src="assets/motivation.png" width=100% >
</div>


## 🔥 Updates
[2025-06-05] 🔥🔥🔥 RefSpatial-Bench is released on [HF](https://huggingface.co/datasets/BAAI/RefSpatial-Bench). Let's evaluate your model's spatial referring ability!


[2025-06-05] RoboRefer is released on [arxiv]().

[2026-06-01] The Project page is set up at [here](https://zhoues.github.io/RoboRefer/).


## 🕶️Overview

### The Overview of RoboRefer

We introduce RoboRefer, **the first 3D-aware reasoning VLM** for multi-step spatial referring with explicit reasoning.

<div align="center"> 
    <img src="assets/pipeline.png" alt="Logo" style="width=100%;vertical-align:middle">
</div>


### The Overview of the RefSpatial Dataset and its Generation Pipeline

We present RefSpatial, a dataset can enable general VLMs to adapt to spatial referring tasks, with **20M QA pairs (2x prior)** and **31 spatial relations (vs. 15 prior)** and **complex reasoning processes (up to 5 steps)**.


<div align="center"> 
    <img src="assets/dataset.png" alt="Logo" style="width=100%;vertical-align:middle">
</div>


## TODO
- [ ] Release RefSpatial-Bench evaluation code (About 1 week).
- [ ] Release the SFT-trained 2B RoboRefer model and inference code (About 2 weeks).
- [ ] Release the SFT-trained 8B RoboRefer model (About 3 weeks).
- [ ] Release the RefSpatial Dataset and SFT training code (About 1 month).
- [ ] Release the RFT-trained RoboRefer model and training code (Maybe 2 months or more).
- [ ] Release the Dataset Generation Pipeline (Maybe 2 months or more).


## Contact
If you have any questions about the code or the paper, feel free to email Enshen (`zhouenshen@buaa.edu.cn`) and Jingkun (`anjingkun02@gmail.com`). 






## Acknowledgment
- This repository is built upon the codebase of [NVILA](https://github.com/NVlabs/VILA), [SpatialRGPT](https://github.com/AnjieCheng/SpatialRGPT) and [R1-V](https://github.com/Deep-Agent/R1-V).

- We acknowledge [OpenImage](https://storage.googleapis.com/openimages/web/index.html), [CA-1M](https://github.com/apple/ml-cubifyanything), [Objaverse](https://github.com/allenai/objaverse-xl), and [Infinigen](https://github.com/princeton-vl/infinigen) for their data and assets.








## 📑 Citation

If you find RoboRefer, RefSpatial, and RefeSpatial-Bench useful for your research, please cite using this BibTeX:
```
TODO
```
