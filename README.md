
<h1 align="center">RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics</h1>

<h3 align="center">From words to exactly where you mean — with RoboRefer</h3>


<p align="center">
  <a href="https://arxiv.org/abs/2506.04308"><img src="https://img.shields.io/badge/arXiv-2506.04308-b31b1b.svg" alt="arXiv"></a>
  &nbsp;
  <a href="https://zhoues.github.io/RoboRefer/"><img src="https://img.shields.io/badge/%F0%9F%8F%A0%20Project-Homepage-blue" alt="Project Homepage"></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/BAAI/RefSpatial-Bench"><img src="https://img.shields.io/badge/🤗%20Benchmark-RefSpatial--Bench-green.svg" alt="Benchmark"></a>
  &nbsp;
  <a href="#"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-Stay%20tuned-yellow" alt="Weights"></a>
</p>

<br>

<div style="text-align: center; background-color: white;">
    <img src="assets/motivation.png" width=100% >
</div>


## 🔥 Updates

[2025-06-23] 🔥🔥🔥 We release the SFT-trained 2B RoboRefer model and inference code with RefSpatial-Bench evaluation code.

[2025-06-06] RefSpatial-Bench is released on [HF](https://huggingface.co/datasets/BAAI/RefSpatial-Bench). Let's evaluate your model's spatial referring ability!

[2025-06-06] RoboRefer is released on [arxiv](https://arxiv.org/abs/2506.04308) and the project page is set up at [here](https://zhoues.github.io/RoboRefer/).



## 🤗 Model Zoo &  Dataset & Benchmark


<table>
  <tr>
    <th>Model/Dataset/Benchmark Name</th>
    <th>HF Path</th>
    <th>Note</th>
  </tr>
  <tr>
    <td>RoboRefer-2B-Depth-Align</td>
    <td><a href="https://huggingface.co/Zhoues/RoboRefer-2B-Depth-Align">2b-depth-align</a></td>
    <td> The 1st SFT step of the 2B model for depth alignment. </td>
  </tr>
    <tr>
    <td>RoboRefer-2B-SFT</td>
    <td><a href="https://huggingface.co/Zhoues/RoboRefer-2B-SFT">2b-sft</a></td>
    <td> The 2nd SFT step of the 2B model for spatial understanding and referring.</td>
  </tr>
  <tr>
    <td>RoboRefer-8B-Depth-Align</td>
    <td>Coming soon</td>
    <!-- <td><a href="https://huggingface.co/Zhoues/RoboRefer-8B-Depth-Align">8b-depth-align</a></td> -->
    <td> The 1st SFT step of the 2B model for depth alignment. </td>
  </tr>
    <tr>
    <td>RoboRefer-8B-SFT</td>
    <td>Coming soon</td>
    <!-- <td><a href="https://huggingface.co/Zhoues/RoboRefer-8B-SFT">8b-sft</a></td> -->
    <td> The 2nd SFT step of the 2B model for spatial understanding and referring.</td>
  </tr>
  <tr>
    <td>RefSpatial Dataset</td>
    <td>Coming soon</td>
    <td> The dataset for spatial understanding and referring with reasoning. </td>
  </tr>
  <tr>
    <td>RefSpatial-Bench</td>
    <td><a href="https://huggingface.co/datasets/BAAI/RefSpatial-Bench">refspatial-bench</a></td>
    <td> The benchmark for spatial referring with reasoning. </td>
  </tr>
</table>

## 🚀 Quick Start
1. Install [Anaconda Distribution](https://www.anaconda.com/download/).
2. Install the necessary Python packages in the environment.
      ```bash
      bash env_step.sh roborefer
      ```
3. Activate a conda environment.
      ```bash
      conda activate roborefer
      ```


## 💡 Inference

1. Download the model weights from the [model zoo](#-model-zoo---dataset--benchmark) (e.g., `roborefer-2b-sft-latest`).

2. Download the relative depth estimation model weights (e.g., [`Depth-Anything-V2-Large`](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true)).

3. Run the inference api server.
      ```bash
      cd API 
      
      python api.py \
      --port 25547 \
      --depth_model_path /your/custom/path/depth_anything_v2_vitl.pth \
      --vlm_model_path /your/custom/path/to/roborefer
      ```

4. Run the inference script with the API and check the results in the `assets` folder.
      ```bash
      cd API 
      
      python use_api.py \
      --image_path ../assets/test.jpg \
      --prompt "Pick the apple in front of the logo side of the leftmost cup." \
      --output_path ../assets/my_result_1.jpg \
      --url http://127.0.0.1:25547

      python use_api.py \
      --image_path ../assets/test.jpg \
      --prompt "Point out the apple nearest to the second cup from left to right." \
      --output_path ../assets/my_result_2.jpg \
      --url http://127.0.0.1:25547

      python use_api.py \
      --image_path ../assets/test.jpg \
      --prompt "Point to the free area between the farthest apple and pink cake." \
      --output_path ../assets/my_result_3.jpg \
      --url http://127.0.0.1:25547
      ```

Below are the results of the inference as examples.

<table>
  <tr>
    <th>Original Image</th>
    <th>"Pick the apple in front of the logo side of the leftmost cup."</th>
    <th>"Point out the apple nearest to the second cup from left to right."</th>
    <th>"Point to the free area between the farthest apple and pink cake."</th>
  </tr>
  <tr>
    <td><img src="assets/test.jpg" width=100% ></td>
    <td><img src="assets/test_result_1.jpg" width=100% ></td>
    <td><img src="assets/test_result_2.jpg" width=100% ></td>
    <td><img src="assets/test_result_3.jpg" width=100% ></td>
</table>





## 🔍 Evaluation for RefSpatial-Bench

1. Open the `Evaluation` folder and download the RefSpatial-Bench dataset from the [model zoo](#-model-zoo---dataset--benchmark).
    ```bash
    cd Evaluation
    git lfs install
    git clone https://huggingface.co/datasets/BAAI/RefSpatial-Bench
    ```

2. Run the API server as the same as the third step in [Inference](#-inference).
    ```bash
    cd API
    python api.py \
    --port 25547 \
    --depth_model_path /your/custom/path/depth_anything_v2_vitl.pth \
    --vlm_model_path /your/custom/path/to/roborefer
    ```

3. Run the evaluation script. 
    - If the `model_name` has `Depth` in the name, the depth model will be used. Therefore, you can choose `RoboRefer-2B-SFT`, `RoboRefer-2B-SFT-Depth` as the model name for RGB-only or RGB-D inference, respectively.
    - The `task_name` can be `Location`, `Placement`, `Unseen`, or `all` to evaluate on all tasks.

    ```bash
    cd Evaluation
    python test_benchmark.py \
    --model_name RoboRefer-2B-SFT-Depth \ 
    --task_name Location \
    --url http://127.0.0.1:25547
    ```

4. Summarize the results.
    - The `model_name` must be the same as the one used in the evaluation script.
    - The `task_name` can be `Location`/`Placement`/`Unseen` to summarize the results for the corresponding task.

    ```bash
    cd Evaluation
    python summarize_acc.py \
    --model_name RoboRefer-2B-SFT-Depth \
    --task_name Location
    ```


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
- [x] Release RefSpatial-Bench evaluation code (About 1 week).
- [x] Release the SFT-trained 2B RoboRefer model and inference code (About 2 weeks).
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
@article{zhou2025roborefer,
  title={RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics},
  author={Zhou, Enshen and An, Jingkun and Chi, Cheng and Han, Yi and Rong, Shanyu and Zhang, Chi and Wang, Pengwei and Wang, Zhongyuan and Huang, Tiejun and Sheng, Lu and others},
  journal={arXiv preprint arXiv:2506.04308},
  year={2025}
}
```
