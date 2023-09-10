# ICCV2023 - Anomaly Detection under Distribution Shift
Official PyTorch implementation of the paper “Anomaly Detection under Distribution Shift”

## Citation
Please cite this paper if it helps your research:
```markdown
@article{cao2023anomaly,
  title={Anomaly Detection under Distribution Shift},
  author={Cao, Tri and Zhu, Jiawen and Pang, Guansong},
  journal={arXiv preprint arXiv:2303.13845},
  year={2023}
}
```
## Dataset
### Download dataset: <br>
-MNIST will be download by torchvision. <br>
-MNIST-M: https://www.kaggle.com/datasets/aquibiqbal/mnistm <br>
-PACS: https://www.kaggle.com/datasets/nickfratto/pacs-dataset <br>
-MVTEC: https://www.mvtec.com/company/research/datasets/mvtec-ad <br>
-CIFAR-10: https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders <br>

### Generate corrupted test set for MVTEC and CIFAR-10
*Change the path of the original data and the generated data if needed.* <br>

To generate currupted data for MVTEC: 
> python generate_corrupted_mvtec.py

To generate currupted data for CIFAR-10: 
> python generate_corrupted_cifar10.py






