
# 3-D Detector for Occluded Object Under Obstructed Conditions
This is a improved version of [3ONet](https://ieeexplore.ieee.org/document/10183841) 
This code is mainly based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and [SA-SSD](https://github.com/skyhehe123/SA-SSD), some codes are from 
[BtcDet](https://github.com/Xharlie/BtcDet) and [PCN](https://github.com/qinglew/PCN-PyTorch)
## Detection Framework
The overall detection framework is shown below.
(1) 3D Sparse Convolution backbone; (2) Point Segmentation Network; 
(3) Point Reconstruction Network,
(4) Fusion and Refinement Network.
We use sparse backbone to efficiently extract multiscale features. Point Segmentation Network and RPN provide valuable information of object'shape. 3ONet applies encoder–decoder
approach for the Point Reconstruction Network to recover the missing shape of the object in the 3D scenes. Fusion and Refinement Network aggregate the instance-level invariant features for proposal refinement.
 
![](./tools/images/framework.png)

## Model Zoo
We release 2 models, which are based on LiDAR-only. 
* All models are trained with 2 RTX 3090 GPUs and are available for download. 

* The models are trained with train split (3712 samples) of KITTI dataset

* The results are the 3D AP(R40) of Car on the *val* set of KITTI dataset.

* These models are not suitable to directly report results on KITTI test set, please use slightly lower score threshold and train the models on all or 80% training data to achieve a desirable performance on KITTI test set.

|                                             |Modality|GPU memory of training| Easy | Mod. | Hard  | download | 
|---------------------------------------------|----------:|----------:|:-------:|:-------:|:-------:|:---------:|
| [3ONet_1](tools/cfgs/models/kitti/3ONet-1.yaml)|LiDAR|~14 GB |94.24 |87.32| 84.17|:---------:|
| [3ONet_2](tools/cfgs/models/kitti/3ONet-2.yaml)|LiDAR|~14 GB| 93.55 |86.24 |83.29 |:---------:|

## Getting Started
### Dependency
Our released implementation is tested on.
+ Ubuntu 20.04
+ Python 3.7.13
+ PyTorch 1.7.1
+ Spconv 1.2.1
+ NVIDIA CUDA 11.1
+ 2x RTX 3090 GPUs

### Prepare dataset

Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):

```
TED
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

You need creat a 'velodyne_depth' dataset to run our multimodal detector:
You can download our preprocessed data from [google (13GB)](https://drive.google.com/file/d/1xki9v_zsQMM8vMVNo0ENi1Mh_GNMjHUg/view?usp=sharing), [baidu (a20o)](https://pan.baidu.com/s/1OH4KIVoSSH7ea3-3CqkZRQ), or generate the data by yourself:
* [Install this project](#installation).
* Download the PENet depth completion model [here (500M)](https://drive.google.com/file/d/1RDdKlKJcas-G5OA49x8OoqcUDiYYZgeM/view?usp=sharing) and put it into ```tools/PENet```.
* Then run the following code to generate RGB pseudo points.
```
cd tools/PENet
python3 main.py --detpath [your path like: ../../data/kitti/training]
```

After 'velodyne_depth' generation, run following command to creat dataset infos:
```
cd ../..
python3 -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
python3 -m pcdet.datasets.kitti.kitti_dataset_mm create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

Anyway, the data structure should be: 
```
TED
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & velodyne_depth
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2 & velodyne_depth
│   │   │── gt_database
│   │   │── gt_database_mm
│   │   │── kitti_dbinfos_train_mm.pkl
│   │   │── kitti_dbinfos_train.pkl
│   │   │── kitti_infos_test.pkl
│   │   │── kitti_infos_train.pkl
│   │   │── kitti_infos_trainval.pkl
│   │   │── kitti_infos_val.pkl
├── pcdet
├── tools
```

### Installation

```
git clone https://github.com/hailanyi/TED.git
cd TED
python3 setup.py develop
```

### Training

Single GPU train:
```
cd tools
python3 train.py --cfg_file ${CONFIG_FILE}
```
For example, if you train the TED-S model:
```
cd tools
python3 train.py --cfg_file cfgs/models/kitti/TED-S.yaml
```

Multiple GPU train: 

You can modify the gpu number in the dist_train.sh and run
```
cd tools
sh dist_train.sh
```
The log infos are saved into log.txt
You can run ```cat log.txt``` to view the training process.

### Evaluation

```
cd tools
python3 test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

For example, if you test the TED-S model:

```
cd tools
python3 test.py --cfg_file cfgs/models/kitti/TED-S.yaml --ckpt TED-S.pth
```

Multiple GPU test: you need modify the gpu number in the dist_test.sh and run
```
sh dist_test.sh 
```
The log infos are saved into log-test.txt
You can run ```cat log-test.txt``` to view the test results.

## License

This code is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

[SA-SDD](https://github.com/skyhehe123/SA-SSD)

[OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

[BtcDet](https://github.com/Xharlie/BtcDet)

[PCN](https://github.com/qinglew/PCN-PyTorch)

## Citation
@article{hoang20233onet,
  title={3ONet: 3D Detector for Occluded Object under Obstructed Conditions},
  author={Hoang, Hiep Anh and Yoo, Myungsik},
  journal={IEEE Sensors Journal},
  year={2023},
  publisher={IEEE}
}





