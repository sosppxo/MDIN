# 3D-GRES: Generalized 3D Referring Expression Segmentation
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-%23EE4C2C.svg?style=&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.7%20|%203.8%20|%203.9-blue.svg?style=&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/)

**[🔗[arXiv]](https://arxiv.org/abs/2407.20664)** &emsp; 
**[📄[PDF]](https://arxiv.org/pdf/2407.20664)** &emsp;


NEWS:🔥3D-GRES is accepted at ACM MM 2024 (Oral)!🔥

Changli Wu, Yihang Liu, Jiayi Ji, Yiwei Ma, Haowei Wang, Gen Luo, Henghui Ding, Xiaoshuai Sun, Rongrong Ji
<img src="docs\3d-gres.png"/>

<img src="docs\mdin.png"/>

## Introduction

3D Referring Expression Segmentation (3D-RES) is dedicated to segmenting a specific instance within a 3D space based on a natural language description.However, current approaches are limited to segmenting a single target, restricting the versatility of the task. To overcome this limitation, we introduce Generalized 3D Referring Expression Segmentation (3D-GRES), which extends the capability to segment any number of instances based on natural language instructions.In addressing this broader task, we propose the Multi-Query Decoupled Interaction Network (MDIN), designed to break down multi-object segmentation tasks into simpler, individual segmentations.MDIN comprises two fundamental components: Text-driven Sparse Queries (TSQ) and Multi-object Decoupling Optimization (MDO). TSQ generates sparse point cloud features distributed over key targets as the initialization for queries. Meanwhile, MDO is tasked with assigning each target in multi-object scenarios to different queries while maintaining their semantic consistency. To adapt to this new task, we build a new dataset, namely Multi3DRes. Our comprehensive evaluations on this dataset demonstrate substantial enhancements over existing models, thus charting a new path for intricate multi-object 3D scene comprehension.

## Installation

Requirements

- Python 3.7 or higher
- Pytorch 1.12
- CUDA 11.3 or higher

The following installation suppose `python=3.8` `pytorch=1.12.1` and `cuda=11.3`.
- Create a conda virtual environment

  ```
  conda create -n 3d-gres python=3.8
  conda activate 3d-gres
  ```

- Clone the repository

  ```
  git clone https://github.com/sosppxo/MDIN.git
  ```

- Install the dependencies

  Install [Pytorch 1.12.1](https://pytorch.org/)

  ```
  pip install spconv-cu113
  conda install pytorch-scatter -c pyg
  pip install -r requirements.txt
  ```

  Install segmentator from this [repo](https://github.com/Karbo123/segmentator) (We wrap the segmentator in ScanNet).

- Setup, Install mdin and pointgroup_ops.

  ```
  sudo apt-get install libsparsehash-dev
  python setup.py develop
  cd gres_model/lib/
  python setup.py develop
  ```

## Data Preparation

### ScanNet v2 dataset

Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

Put the downloaded `scans` folder as follows.

```
MDIN
├── data
│   ├── scannetv2
│   │   ├── scans
```

Split and preprocess point cloud data

```
cd data/scannetv2
bash prepare_data.sh
```

The script data into train/val folder and preprocess the data. After running the script the scannet dataset structure should look like below.

```
MDIN
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── train
│   │   ├── val
```

### ScanRefer dataset
Download [ScanRefer](https://github.com/daveredrum/ScanRefer) annotations following the instructions.

In the original ScanRefer annotations, all `ann_id` within each scene were individually assigned based on the corresponding `object_id`, resulting in duplicate `ann_id`. We have modified the ScanRefer annotations, and the revised annotation data, where each `ann_id` within a scene is unique, can be accessed [here](https://stuxmueducn-my.sharepoint.com/:f:/g/personal/liuyihang_stu_xmu_edu_cn/Euro_ViDA6JPiF05ayQqa_EB32qVgfhxc0c0-Oy5S7XjrA?e=0t4VlB).

Put the downloaded `ScanRefer` folder as follows.
```
MDIN
├── data
│   ├── ScanRefer
│   │   ├── ScanRefer_filtered_train_new.json
│   │   ├── ScanRefer_filtered_val_new.json
```
### Multi3DRefer dataset
Downloading the [Multi3DRefer](https://aspis.cmpt.sfu.ca/projects/multi3drefer/data/multi3drefer_train_val.zip) annotations. 

Put the downloaded `Multi3DRefer` folder as follows.
```
MDIN
├── data
│   ├── Multi3DRefer
│   │   ├── multi3drefer_train.json
│   │   ├── multi3drefer_val.json
```

## Pretrained Backbone

Download [SPFormer](https://stuxmueducn-my.sharepoint.com/:f:/g/personal/22920182204313_stu_xmu_edu_cn/Em7yJHaCHAxFpM15uVwk9cgByDp-67lWQg59vkU-zokHYA?e=IuZV0D) pretrained model (We only use the Sparse 3D U-Net backbone for training).

Move the pretrained model to backbones.
```
mkdir backbones
mv ${Download_PATH}/sp_unet_backbone.pth backbones/
```

## Models
Download pretrain models and move it to checkpoints.
|Benchmark | Task  | mIoU | Acc@0.25 | Acc@0.5 | Model |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Multi3DRes | 3D-GRES | 47.5 | 66.9 | 44.7 | [Model](https://stuxmueducn-my.sharepoint.com/:u:/g/personal/liuyihang_stu_xmu_edu_cn/EUJsjxlCnThPhhDd9qU7uzkBQh6EsTc8h4TeREr3qbxKDw?e=igqKF3) |
| ScanRefer   | 3D-RES | 48.3 | 58.0 | 53.1 | [Model](https://stuxmueducn-my.sharepoint.com/:u:/g/personal/liuyihang_stu_xmu_edu_cn/Ecjim2JRxFZAvVitoKVqFOwByVfbJoQRN62O4xA2kTNsuw?e=kgAClI) |
| Nr3D        | 3D-RES | 38.6 | 48.4 | 42.2 | [Model](https://stuxmueducn-my.sharepoint.com/:u:/g/personal/liuyihang_stu_xmu_edu_cn/EdQoF63xEodGpL8TLktE-WABbd3dcrlsGjP0lnooWzMfOg?e=2QLyDQ) |
| Sr3D        | 3D-RES | 46.4 | 56.6 | 51.3 | [Model](https://stuxmueducn-my.sharepoint.com/:u:/g/personal/liuyihang_stu_xmu_edu_cn/EZ4cZn-dNQpIoj1ZF9Zw-MMBuHfv_LGEZMXaeePT1sdX5A?e=RbS3LT) |


## Training
For 3D-GRES:
```
bash scripts/train_3dgres.sh
```
For 3D-RES:
```
bash scripts/train_3dres.sh
```

## Inference
For 3D-GRES:
```
bash scripts/test_3dgres.sh
```
For 3D-RES:
```
bash scripts/test_3dres.sh
```
## Citation

If you find this work useful in your research, please cite:

```
@misc{wu20243dgresgeneralized3dreferring,
      title={3D-GRES: Generalized 3D Referring Expression Segmentation}, 
      author={Changli Wu and Yihang Liu and Jiayi Ji and Yiwei Ma and Haowei Wang and Gen Luo and Henghui Ding and Xiaoshuai Sun and Rongrong Ji},
      year={2024},
      eprint={2407.20664},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.20664}, 
}
```

## Ancknowledgement

Sincerely thanks for [ReLA](https://github.com/henghuiding/ReLA), [M3DRef-CLIP](https://github.com/3dlg-hcvc/M3DRef-CLIP), [EDA](https://github.com/yanmin-wu/EDA), [SceneGraphParser](https://github.com/vacancy/SceneGraphParser), [SoftGroup](https://github.com/thangvubk/SoftGroup), [SSTNet](https://github.com/Gorilla-Lab-SCUT/SSTNet) and [SPFormer](https://github.com/sunjiahao1999/SPFormer) repos. This repo is build upon them.
