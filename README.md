# PlanarRecon: Real-time 3D Plane Detection and Reconstruction from Posed Monocular Videos
### [Project Page](https://neu-vi.github.io/planarrecon) | [Paper](https://arxiv.org/pdf/2206.07710.pdf)
<br/>

> PlanarRecon: Real-time 3D Plane Detection and Reconstruction from Posed Monocular Videos  
> [Yiming Xie](https://ymingxie.github.io), [Matheus Gadelha](http://mgadelha.me/), [Fengting Yang](http://personal.psu.edu/fuy34/), [Xiaowei Zhou](https://xzhou.me/), [Huaizu Jiang](https://jianghz.me/)  
> CVPR 2022

![real-time video](assets/planarrecon_demo.gif)

<br/>


## How to Use

### Installation
```shell
conda env create -f environment.yaml
conda activate planarrecon
```
Follow instructions in [torchsparse](https://github.com/mit-han-lab/torchsparse) to install torchsparse.  

### Pretrained Model on ScanNet
Download the [pretrained weights](https://drive.google.com/file/d/1XLL5X2M5BPo89An4jom5s0zhQOiyS_h8/view?usp=sharing) and put it under 
`PROJECT_PATH/checkpoints/release`.
You can also use [gdown](https://github.com/wkentaro/gdown) to download it in command line:
```bash
gdown --id 1XLL5X2M5BPo89An4jom5s0zhQOiyS_h8
```

### Data Preperation for ScanNet
Download and extract ScanNet by following the instructions provided at http://www.scan-net.org/.
<details>
  <summary>[Expected directory structure of ScanNet (click to expand)]</summary>
  
You can obtain the train/val/test split information from [here](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark).
```
DATAROOT
└───scannet
│   └───scans
│   |   └───scene0000_00
│   |       └───color
│   |       │   │   0.jpg
│   |       │   │   1.jpg
│   |       │   │   ...
│   |       │   ...
│   └───scans_raw
│   |   └───scene0000_00
│   |       └───scene0000_00.aggregation.json
│   |       └───scene0000_00_vh_clean_2.labels.ply
│   |       └───scene0000_00_vh_clean_2.0.010000.segs.json
│   |       │   ...
|   └───scannetv2_test.txt
|   └───scannetv2_train.txt
|   └───scannetv2_val.txt
|   └───scannetv2-labels.combined.tsv
```
</details>

Next run the data preparation script which parses the raw data format into the processed pickle format.
This script also generates the ground truth Planes.
The plane generation code is modified from [PlaneRCNN](https://github.com/NVlabs/planercnn/blob/master/data_prep/parse.py).

<details>
  <summary>[Data preparation script]</summary>

```bash
# Change PATH_TO_SCANNET accordingly.
# For the training/val split:
python tools/generate_gt.py --data_path PATH_TO_SCANNET --save_name planes_9/ --window_size 9 --n_proc 2 --n_gpu 1
```
</details>


### Inference on ScanNet val-set
```bash
python main.py --cfg ./config/test.yaml
```

The planes will be saved to `PROJECT_PATH/results`.


### Evaluation on ScanNet val-set
Evaluate 3D geometry:
```
python tools/eval3d_geo_ins.py --model ./results/scene_scannet_release_68 --n_proc 16
```

Evaluate plane segmentation:
```
# generate gt instance txt
python tools/prepare_inst_gt_txt.py --val_list path_to_scannetv2_val.txt --plane_mesh_path path_to_planes_tsdf_9

# eval instance
python tools/eval3d_instance.py --pred_path path_to_pred/plane_ins --gt_path path_to_planes_tsdf_9/instance --scan_list path_to_scannetv2_val.txt
```

### Training on ScanNet

Similar to NeuralRecon, the training is seperated to three phases and the switching is controlled manually for now:

-  Phase 0 (the first 0-20 epoch), training single fragments.  
`MODEL.FUSION.FUSION_ON=False, MODEL.TRACKING=False`
```bash
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 main.py --cfg ./config/train_phase0.yaml
```
- Phase 1 (21-35 epoch), with `GRUFusion`.  
`MODEL.FUSION.FUSION_ON=True, MODEL.TRACKING=False`
```bash
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 main.py --cfg ./config/train_phase1.yaml
```
- Phase 2 (the remaining 35-50 epoch), with `Matching/Fusion`.  
`MODEL.FUSION.FUSION_ON=True, MODEL.TRACKING=True`
```bash
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch --nproc_per_node=2 main.py --cfg ./config/train_phase2.yaml
```



### Real-time Demo on Custom Data with Camera Poses from ARKit.
We provide a demo of PlanarRecon running with self-captured ARKit data. Please refer to [DEMO.md](DEMO.md) for details.
We also provide the [example data](https://drive.google.com/file/d/1FKccOUCW2T_rV81VhqVeqeo-dec8ooNW/view?usp=sharing) captured using iPhoneXR.
Incrementally saving and visualizing are not enabled in PlanarRecon for now.


## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@article{xie2022planarrecon,
  title={{PlanarRecon}: Real-Time {3D} Plane Detection and Reconstruction from Posed Monocular Videos},
  author={Xie, Yiming and Gadelha, Matheus and Yang, Fengting and Zhou, Xiaowei and Jiang, Huaizu},
  journal={CVPR},
  year={2022}
}
```

## Acknowledgment
Some of the code and installation guide in this repo is borrowed from [NeuralRecon](https://github.com/zju3dv/NeuralRecon)! 
We also thank [Atlas](https://github.com/magicleap/Atlas) for the 3D geometry evaluation. 
