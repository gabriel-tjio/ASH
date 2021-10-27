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
CUDA_VISIBLE_DEVICES=0 python CLAN_train.py --snapshot-dir ./snapshots/GTA2Cityscapes
```

### Evaluate
```
CUDA_VISIBLE_DEVICES=0 python CLAN_evaluate.py --restore-from  ./snapshots/GTA2Cityscapes/GTA5_100000.pth --save ./result/GTA2Cityscapes_100000
```


### Compute IoU
```
python CLAN_iou.py ./data/Cityscapes/gtFine/val result/GTA2Cityscapes_100000
```

#### Tip: The best-performance model might not be the final one in the last epoch. If you want to evaluate every saved models in bulk, please use CLAN_evaluate_bulk.py and CLAN_iou_bulk.py, the result will be saved in an Excel sheet.
```
CUDA_VISIBLE_DEVICES=0 python CLAN_evaluate_bulk.py
python CLAN_iou_bulk.py
```

#### This code is heavily borrowed from the baseline CLAN https://github.com/RoyalVane/CLAN ) and pytorch_adain (https://github.com/naoto0804/pytorch-AdaIN)

GTA trained model
https://drive.google.com/file/d/1eVnDMC3ytyl5Wx8H5VPKoTNq1UGXmS56/view?usp=sharing

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

