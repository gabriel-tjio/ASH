# Adversarial Semantic Hallucination for Domain Generalized Semantic Segmentation
This is a [pytorch](http://pytorch.org/) implementation of [ASH]().



### Prerequisites
- Python 3.6
- GPU Memory >= 16G
- Pytorch 1.6.0

### Getting started

- Download [The GTA5 Dataset]( https://download.visinf.tu-darmstadt.de/data/from_games/ )

- Download [The SYNTHIA Dataset]( http://synthia-dataset.net/download-2/ )

- Download [The Cityscapes Dataset]( https://www.cityscapes-dataset.com/ )



The data folder is structured as follows:
```
├── data/
│   ├── Cityscapes/     
|   |   ├── gtFine/
|   |   ├── leftImg8bit/
│   ├── GTA5/
|   |   ├── images/
|   |   ├── labels/
│   ├── SYNTHIA/ 
|   |   ├── RAND_CITYSCAPES/
│   └── 			
└── model/
│   ├── DeepLab_resnet_pretrained.pth
...
```

### Train
```
CUDA_VISIBLE_DEVICES=0 python train.py --snapshot-dir ./snapshots/GTA2Cityscapes
```



#### This code is heavily borrowed from the baseline CLAN https://github.com/RoyalVane/CLAN ) and pytorch_adain (https://github.com/naoto0804/pytorch-AdaIN)

GTA trained model
https://drive.google.com/file/d/1eVnDMC3ytyl5Wx8H5VPKoTNq1UGXmS56/view?usp=sharing

SYN trained model
https://drive.google.com/file/d/1XTmJhXCGeD2KrK7M7Xklq0Gayvd0n86e/view?usp=sharing

### Citation
If you use this code in your research please consider citing



```
@inproceedings{ASH,
  author    = {Gabriel Tjio and
               Ping Liu and
               Joey Tianyi Zhou and
               Rick Siow Mong Goh},
  title     = {Adversarial Semantic Hallucination for Domain Generalized Semantic
               Segmentation},
 booktitle = {IEEE Winter Conf. on Applications of Computer Vision},
 year = {2022}
}

