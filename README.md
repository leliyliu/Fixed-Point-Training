# Fixed-Point-Training
Low-bit &amp; Hardware-aware Quantization Training 

## Experiment Results

### CIFAR10

| Methods | Models | Acc | Precision | Dataset | 
| - | - | - | - | - | 
| FP | ResNet-20 | 92.32 | fp32 | CIFAR10 |
| FP8 | ResNet-20 | 92.21 | fp8 | CIFAR10 |
| Unified | ResNet-20 | 91.95 | int8 | CIFAR10 |
| Distributed | ResNet-20 | <b>92.76</b> | int8 | CIFAR10 | 

| Methods | Models | Acc | Precision | Dataset | 
| - | - | - | - | - | 
| FP | MobileNetV2 | 94.39 | fp32 | CIFAR10 |
| DoReFa | MobileNetV2 | 91.03 | int8 | CIFAR10 |
| WAGEUBN | MobileNetV2 | 92.32 | int8 | CIFAR10 |
| SBM | MobileNetV2 | 93.57 | int8 | CIFAR10 |
| CPT | MobileNetV2 | 93.76 | int8 | CIFAR10 |
| Unified | MobileNetV2 | 93.38 | int8 | CIFAR10 |
| Distributed | MobileNetV2 | <b>94.37</b> | int8 | CIFAR10 | 

| Methods | Models | Acc | Precision | Dataset | 
| - | - | - | - | - | 
| FP | InceptionV3 | 94.89 | fp32 | CIFAR10 |
| Unified | InceptionV3 | 95.00 | int8 | CIFAR10 |
| Distributed | InceptionV3 | <b>95.21</b> | int8 | CIFAR10 | 


### CIFAR100

| Methods | Models | Acc | Precision | Dataset | 
| - | - | - | - | - | 
| DoReFa | MobileNetV2 | 70.17 | int8 | CIFAR100 |
| WAGEUBN | MobileNetV2 | 71.45 | int8 | CIFAR100 |
| SBM | MobileNetV2 | 75.28 | int8 | CIFAR100 |
| CPT | MobileNetV2 | <b>75.65</b> | int8 | CIFAR100 |

| Methods | Models | Acc | Precision | Dataset | 
| - | - | - | - | - | 
| DoReFa | ResNet-74 | 69.31 | int8 | CIFAR100 |
| WAGEUBN | ResNet-74 | 69.61 | int8 | CIFAR100 |
| SBM | ResNet-74 | 71.44 | int8 | CIFAR100 |
| CPT | ResNet-74 | <b>72.35</b> | int8 | CIFAR100 |

### ImageNet



## Todo-List
- [ ] full-precision baseline for CIFAR10/CIFAR100
- [ ] A Simple int8 quantization training Framework
- [ ] CPT baseline for CIFAR10/CIFAR100 