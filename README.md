
<h1 align="center">RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics</h1>

<h3 align="center">From words to exactly where you mean — with RoboRefer</h3>


<p align="center">
  <a href="https://arxiv.org/abs/2506.04308"><img src="https://img.shields.io/badge/arXiv-2506.04308-b31b1b.svg" alt="arXiv"></a>
  &nbsp;
  <a href="https://zhoues.github.io/RoboRefer/"><img src="https://img.shields.io/badge/%F0%9F%8F%A0%20Project-Homepage-blue" alt="Project Homepage"></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/BAAI/RefSpatial-Bench"><img src="https://img.shields.io/badge/🤗%20Benchmark-RefSpatial--Bench-green.svg" alt="Benchmark"></a>
  &nbsp;
  <a href="https://huggingface.co/collections/Zhoues/roborefer-and-refspatial-6857c97848fab02271310b89"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-RoboRefer-yellow" alt="Weights"></a>
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
    <th>Model/Dataset/Benchmark</th>
    <th>Note</th>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/Zhoues/NVILA-2B-Depth">NVILA-2B-Depth</a></td>
    <td> The base model with depth encoder initialized from image encoder. </td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/Zhoues/RoboRefer-2B-Depth-Align">RoboRefer-2B-Align</a></td>
    <td> The 1st SFT step of the 2B model for depth alignment. </td>
  </tr>
    <tr>
    <td><a href="https://huggingface.co/Zhoues/RoboRefer-2B-SFT">RoboRefer-2B-SFT</a></td>
    <td> The 2nd SFT step of the 2B model for spatial understanding and referring.</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/Zhoues/NVILA-8B-Depth">NVILA-8B-Depth</a></td>
    <td> The base model with depth encoder initialized from image encoder. </td>
  </tr>
  <tr>
    <td>RoboRefer-8B-Align (Coming soon)</td>
    <td> The 1st SFT step of the 8B model for depth alignment. </td>
  </tr>
  <tr>
    <td>RoboRefer-8B-SFT (Coming soon)</td>
    <td> The 2nd SFT step of the 8B model for spatial understanding and referring.</td>
  </tr>
  <tr>
    <td>RoboRefer-2B-RFT (Coming soon)</td>
    <td> The RFT-trained 2B model for multi-step spatial referring with reasoning.</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/datasets/JingkunAn/RefSpatial">RefSpatial Dataset</a></td>
    <td> The dataset for spatial understanding and referring with reasoning. </td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/datasets/BAAI/RefSpatial-Bench">RefSpatial-Bench</a></td>
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

1. Download the model weights from the [model zoo](#-model-zoo---dataset--benchmark) (e.g., `RoboRefer-2B-SFT`).

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
    - If the `model_name` has `Depth` in the name, the depth model will be used. Therefore, you can choose `RoboRefer-2B-SFT`, `RoboRefer-2B-SFT-Depth` as the model name for RGB/RGB-D inference, respectively.
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

## 📚 Training

1. Download the RefSpatial-Bench dataset from the [model zoo](#-model-zoo---dataset--benchmark) and decompress the dataset. The provided `unzip_dataset.sh` script could decompress all of the `*.tar.gz` files. Please run it from the `RefSpatial` root directory.

      ```bash
      cd RefSpatial
      bash unzip_dataset.sh
      ```
      <div style="background-color: #eff6ff; border-left: 4px solid #3b82f6; padding: 0.75em 1em; margin-top: 1em; color: #1e3a8a; font-weight: bold; border-radius: 0.375em;">
        💡 Info: The full raw dataset (~412GB) is the same format as the LLaVA dataset.
      </div>
      <br>
      This script will automatically perform the following actions:

      1. **Merge Split Files**: For files that are split into `.part_a`, `.part_b`, etc., the script will use the `cat` command to combine them into a single, complete `.tar.gz` file. For example, `image.tar.gz.part_a`, `...` will be merged into `image.tar.gz`.
      2. **Extract Archives**: The script will then use the `tar` command to extract all `.tar.gz` archives into their current directories.

2. (Optional) Clean Up Archives. If you wish to delete all `.tar.gz` and `.part_*` files after successful decompression to save disk space, you can run:

    ```bash
    bash delete_tar_gz.sh
    ```
    <div style="background-color: #ffe4e6; border-left: 4px solid #dc2626; padding: 0.75em 1em; margin-top: 1em; color: #b91c1c; font-weight: bold; border-radius: 0.375em;">
      ⚠️ Warning: Please run this script only after confirming that all data has been successfully decompressed.
    </div>
    <br>

3. Download the RoboRefer base model weights or depth aligned model weights from the [model zoo](#-model-zoo---dataset--benchmark).

4. Add dataset you want to train on in the `register_datasets_mixtures()` function in `RoboRefer/llava/data/datasets_mixture.py`.

    > To use RefSpatial dataset for model training, you need to match the entries of the image and depth path in the JSON files with the decompressed image and depth map files. Below is the mapping of each JSON file to its corresponding image and depth folders.

      <details>
      <summary>JSON to Image and Depth Folder PathMapping</summary>
      <pre><code>{
        "2D": {
          "folder": "RefSpatial/2D",
          "jsons": {
            "choice_qa.json": {
              "image_root": "RefSpatial/2D/image",
              "depth_root": "RefSpatial/2D/depth"
            },
            "reasoning_template_qa.json": {
              "image_root": "RefSpatial/2D/image",
              "depth_root": "RefSpatial/2D/depth"
            }
          }
        },
        "3D": {
          "folder": "RefSpatial/3D",
          "jsons": {
            "choice_qa.json": {
              "depth_root": "RefSpatial/3D/depth",
              "image_root": "RefSpatial/3D/image"
            },
            "multi_view_qa.json": {
              "depth_root": "RefSpatial/3D/depth_multi_view",
              "image_root": "RefSpatial/3D/image_multi_view"
            },
            "reasoning_template_qa.json": {
              "depth_root": "RefSpatial/3D/depth",
              "image_root": "RefSpatial/3D/image"
            },
            "vacant_qa.json": {
              "depth_root": "RefSpatial/3D/depth",
              "image_root": "RefSpatial/3D/image"
            },
            "visual_choice_qa.json": {
              "depth_root": "RefSpatial/3D/depth",
              "image_root": "RefSpatial/3D/image_visual_choice"
            }
          }
        },
        "Simulator": {
          "folder": "RefSpatial/Simulator",
          "jsons": {
            "metadata.json": {
              "image_root": "RefSpatial/Simulator/image",
              "depth_root": "RefSpatial/Simulator/depth"
            }
          }
        }
      }
      </code></pre>

      </details>

    <br>

    4.1. We designed a flexible `dataset_type` to support both RGB-only and RGB-D training. To train with RGB-D, set the `depth_path` field in the dataset config. For RGB-only training, simply omit the `depth_path`. Below is an example of how to register the RefSpatial dataset for both RGB-only and RGB-D training in the `register_datasets_mixtures()` function in `RoboRefer/llava/data/datasets_mixture.py`. The RefSpatial dataset has already been implemented in its corresponding module.
      <details>
      <summary>Example of Adding RefSpatial Dataset</summary>
      <pre><code>
      def register_datasets_mixtures():

          ### OpenImage (2D Dataset)
          2D_choice_qa = Dataset(
              dataset_name="2D_choice_qa",
              dataset_type="spatialdataset",
              data_path="./RefSpatial/2D/choice_qa.json",
              image_path="./RefSpatial/2D/image",
              depth_path="./RefSpatial/2D/depth"
          )
          add_dataset(2D_choice_qa)

          2D_choice_qa_RGB = Dataset(
              dataset_name="2D_choice_qa_RGB",
              dataset_type="spatialdataset",
              data_path="./RefSpatial/2D/choice_qa.json",
              image_path="./RefSpatial/2D/image"
          )
          add_dataset(2D_choice_qa_RGB)

          2D_reasoning_template_qa = Dataset(
              dataset_name="2D_reasoning_template_qa",
              dataset_type="spatialdataset",
              data_path="./RefSpatial/2D/reasoning_template_qa.json",
              image_path="./RefSpatial/2D/image",
              depth_path="./RefSpatial/2D/depth"
          )
          add_dataset(2D_reasoning_template_qa)

          2D_reasoning_template_qa_RGB = Dataset(
              dataset_name="2D_reasoning_template_qa_RGB",
              dataset_type="spatialdataset",
              data_path="./RefSpatial/2D/reasoning_template_qa.json",
              image_path="./RefSpatial/2D/image"
          )
          add_dataset(2D_reasoning_template_qa_RGB)

          ### CA-1M (3D Dataset)
          3D_choice_qa = Dataset(
              dataset_name="3D_choice_qa",
              dataset_type="spatialdataset",
              data_path="./RefSpatial/3D/choice_qa.json",
              image_path="./RefSpatial/3D/image",
              depth_path="./RefSpatial/3D/depth"
          )
          add_dataset(3D_choice_qa)

          3D_choice_qa_RGB = Dataset(
              dataset_name="3D_choice_qa_RGB",
              dataset_type="spatialdataset",
              data_path="./RefSpatial/3D/choice_qa.json",
              image_path="./RefSpatial/3D/image"
          )
          add_dataset(3D_choice_qa_RGB)

          3D_reasoning_template_qa = Dataset(
              dataset_name="3D_reasoning_template_qa",
              dataset_type="spatialdataset",
              data_path="./RefSpatial/3D/reasoning_template_qa.json",
              image_path="./RefSpatial/3D/image",
              depth_path="./RefSpatial/3D/depth"
          )
          add_dataset(3D_reasoning_template_qa)

          3D_reasoning_template_qa_RGB = Dataset(
              dataset_name="3D_reasoning_template_qa_RGB",
              dataset_type="spatialdataset",
              data_path="./RefSpatial/3D/reasoning_template_qa.json",
              image_path="./RefSpatial/3D/image"
          )
          add_dataset(3D_reasoning_template_qa_RGB)

          3D_vacant_qa = Dataset(
              dataset_name="3D_vacant_qa",
              dataset_type="spatialdataset",
              data_path="./RefSpatial/3D/vacant_qa.json",
              image_path="./RefSpatial/3D/image",
              depth_path="./RefSpatial/3D/depth"
          )
          add_dataset(3D_vacant_qa)

          3D_vacant_qa_RGB = Dataset(
              dataset_name="3D_vacant_qa_RGB",
              dataset_type="spatialdataset",
              data_path="./RefSpatial/3D/vacant_qa.json",
              image_path="./RefSpatial/3D/image"
          )
          add_dataset(3D_vacant_qa_RGB)

          3D_multi_view_qa = Dataset(
              dataset_name="3D_multi_view_qa",
              dataset_type="spatialdataset",
              data_path="./RefSpatial/3D/multi_view_qa.json",
              image_path="./RefSpatial/3D/image_multi_view",
              depth_path="./RefSpatial/3D/depth_multi_view"
          )
          add_dataset(3D_multi_view_qa)

          3D_multi_view_qa_RGB = Dataset(
              dataset_name="3D_multi_view_qa_RGB",
              dataset_type="spatialdataset",
              data_path="./RefSpatial/3D/multi_view_qa.json",
              image_path="./RefSpatial/3D/image_multi_view"
          )
          add_dataset(3D_multi_view_qa_RGB)

          3D_visual_choice_qa = Dataset(
              dataset_name="3D_visual_choice_qa",
              dataset_type="spatialdataset",
              data_path="./RefSpatial/3D/visual_choice_qa.json",
              image_path="./RefSpatial/3D/image_visual_choice",
              depth_path="./RefSpatial/3D/depth"
          )
          add_dataset(3D_visual_choice_qa)

          3D_visual_choice_qa_RGB = Dataset(
              dataset_name="3D_visual_choice_qa_RGB",
              dataset_type="spatialdataset",
              data_path="./RefSpatial/3D/visual_choice_qa.json",
              image_path="./RefSpatial/3D/image_visual_choice"
          )
          add_dataset(3D_visual_choice_qa_RGB)

          ### Simulator (Simulator Dataset)
          simulation_dataset = Dataset(
              dataset_name="simulation_dataset",
              dataset_type="spatialdataset",
              data_path="./RefSpatial/Simulator/metadata.json",
              image_path="./RefSpatial/Simulator/image",
              depth_path="./RefSpatial/Simulator/depth"
          )
          add_dataset(simulation_dataset)

          simulation_dataset_RGB = Dataset(
              dataset_name="simulation_dataset_RGB",
              dataset_type="spatialdataset",
              data_path="./RefSpatial/Simulator/metadata.json",
              image_path="./RefSpatial/Simulator/image"
          )
          add_dataset(simulation_dataset_RGB)
      </code></pre>

      </details>

    <br>
    
    4.2.  In `scripts/RoboRefer`, we provide scripts for depth alignment, SFT training, and RFT training (coming soon). You can run them using the commands below. Be sure to update the base model path and add your custom dataset(s) in the script. After registering your datasets in `register_datasets_mixtures()`, you can use `+` to include multiple datasets.

    ```bash
    bash scripts/roborefer/depth_align_2B.sh # or bash scripts/roborefer/depth_align_2B_cluster.sh. If your use cluster for training, you can run this script. 8B variant is the same.

    bash scripts/roborefer/depth_sft_2B.sh # or bash scripts/roborefer/depth_sft_2B_cluster.sh. If your use cluster for training, you can run this script. 8B variant is the same.
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
- [x] Release the RefSpatial Dataset and SFT training code (About 1 month).
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
