# ICCV2023 - Anomaly Detection under Distribution Shift
Official PyTorch implementation of the ICCV'23 paper “Anomaly Detection under Distribution Shift”

## Environment
Create suitable conda environment:
> conda env create -f environment.yml
## Dataset
### 1. Download dataset: 
-MNIST_grey: https://www.kaggle.com/datasets/jidhumohan/mnist-png <br>
-MNIST_M: https://www.kaggle.com/datasets/aquibiqbal/mnistm <br>
-PACS: https://www.kaggle.com/datasets/nickfratto/pacs-dataset <br>
-MVTEC: https://www.mvtec.com/company/research/datasets/mvtec-ad <br>
-CIFAR-10: https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders <br>

### 2. Generate corrupted test set for MVTEC and CIFAR-10
To generate currupted data for MVTEC: 
> python generate_corrupted_mvtec.py

To generate currupted data for CIFAR-10: 
> python generate_corrupted_cifar10.py
## DINL (for training phase)
To train the model, please run the corresponding file train_namedataset_DINL.py <br>
For example, to train DINL for PACS:
> python train_PACS_DINL.py

## ATTA (for inference phase)
Note: change the path to the normal image for each dataset if needed. <br>
To run the inference, please run the corresponding file inference_namedataset_ATTA.py <br>
For example, to use ATTA for PACS:
> python inference_PACS_ATTA.py

## Citation
The paper is available at [ICCV'23 proceedings](https://openaccess.thecvf.com//content/ICCV2023/html/Cao_Anomaly_Detection_Under_Distribution_Shift_ICCV_2023_paper.html) or [arXiv](https://arxiv.org/abs/2303.13845).

Please cite this paper if it helps your research:
```bibtex
@InProceedings{Cao_2023_ICCV,
    author    = {Cao, Tri and Zhu, Jiawen and Pang, Guansong},
    title     = {Anomaly Detection Under Distribution Shift},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {6511-6523}
}
```







