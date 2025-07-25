# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import dataclass, field


@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default="torch")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    meta_path: str = field(default=None, metadata={"help": "Path to the meta data for webdataset."})
    image_path: str = field(default=None, metadata={"help": "Path to the training image data."})
    # NOTE(Zhouenshen): Add the depth path for spatialdataset
    depth_path: str = field(default=None, metadata={"help": "Path to the training depth data."})
    caption_choice: str = field(default=None, metadata={"help": "Path to the caption directory for recaption."})
    description: str = field(
        default=None,
        metadata={
            "help": "Detailed desciption of where the data is from, how it is labelled, intended use case and the size of the dataset."
        },
    )
    test_script: str = (None,)
    maintainer: str = (None,)
    ############## ############## ############## ############## ############## ##############
    caption_choice: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    caption_choice_2: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    start_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})
    end_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})


DATASETS_LEGACY = {}


def add_dataset(dataset):
    if dataset.dataset_name in DATASETS_LEGACY:
        # make sure the data_name is unique
        warnings.warn(f"{dataset.dataset_name} already existed in DATASETS. Make sure the name is unique.")
    assert "+" not in dataset.dataset_name, "Dataset name cannot include symbol '+'."
    DATASETS_LEGACY.update({dataset.dataset_name: dataset})


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