# S3C
Self-Supervised Stochastic Classifiers for Few-Shot Class-Incremental Learning


### Dependencies
All library details given in FSCIL.yml file. To install FSCIL environment run the following command.
```
$conda env create -f FSCIL.yml
```
### Dataset
We followed same protocol as [CEC](https://github.com/icoz69/CEC-CVPR2021) for downloading datasets.

### Training
#### CIFAR-100
```
$bash run_cifar_s3c.sh
```

For both CUB and miniImageNet, need to change image shape inside training and test functions (CUB 224*224, miniImageNet 84*84)

#### CUB-200
```
$bash run_cub_s3c.sh
```
#### miniImageNet-100
```
$bash run_imagenet_s3c.sh
```
