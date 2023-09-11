# ICCV2023 - Anomaly Detection under Distribution Shift
Official PyTorch implementation of the ICCV'23 paper “Anomaly Detection under Distribution Shift”

## Environment
Create suitable conda environment:
> conda env create -f environment.yml
## Dataset
### 1. Download dataset: <br>
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
For example, to train ATTA for PACS:
> python inference_PACS_ATTA.py

## Citation
Please cite this paper if it helps your research:
```bibtex
@article{cao2023anomaly,
  title={Anomaly Detection under Distribution Shift},
  author={Cao, Tri and Zhu, Jiawen and Pang, Guansong},
  journal={arXiv preprint arXiv:2303.13845},
  year={2023}
}
```







