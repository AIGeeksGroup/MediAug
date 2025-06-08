# MediAug
This is the code repository for the paper:
> **MediAug: Exploring Visual Augmentation in Medical Imaging**
>
> Xuyin Qi\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*<sup>†</sup>, Canxuan Gang\*, Hao Zhang, Lei Zhang, Zhiwei Zhang, and [Yang Zhao](https://yangyangkiki.github.io/)<sup>#</sup>
>
> \*Equal contribution. <sup>†</sup>Project lead. <sup>#</sup>Corresponding author.
>
> ***MIUA 2025***
>
> **[[arXiv]](https://www.arxiv.org/abs/2504.18983)** **[[Paper with Code]](https://paperswithcode.com/paper/mediaug-exploring-visual-augmentation-in)** **[[HF Paper]](https://huggingface.co/papers/2504.18983)**

![image](https://github.com/AIGeeksGroup/MediAug/blob/main/arch.png)


## Citation

If you use any content of this repo for your work, please cite the following our paper:
```
@article{qi2025mediaug,
  title={MediAug: Exploring Visual Augmentation in Medical Imaging},
  author={Qi, Xuyin and Zhang, Zeyu and Gang, Canxuan and Zhang, Hao and Zhang, Lei and Zhang, Zhiwei and Zhao, Yang},
  journal={arXiv preprint arXiv:2504.18983},
  year={2025}
}
```

## Introduction
Data augmentation is essential in medical imaging for improving classification accuracy, lesion detection, and organ segmentation under limited data conditions. However, two significant challenges remain. First, a pronounced domain gap between natural photographs and medical images can distort critical disease features. Second, augmentation studies in medical imaging are fragmented and limited to single tasks or architectures, leaving the benefits of advanced mix-based strategies unclear. To address these challenges, we propose a unified evaluation framework with six mix-based augmentation methods integrated with both convolutional and transformer backbones on brain tumour MRI and eye disease fundus datasets. Our contributions are threefold. (1) We introduce MediAug, a comprehensive and reproducible benchmark for advanced data augmentation in medical imaging. (2) We systematically evaluate MixUp, YOCO, CropMix, CutMix, AugMix, and SnapMix with ResNet-50 and ViT-B backbones. (3) We demonstrate through extensive experiments that MixUp yields the greatest improvement on the brain tumor classification task for ResNet-50 with 79.19% accuracy and SnapMix yields the greatest improvement for ViT-B with 99.44% accuracy, and that YOCO yields the greatest improvement on the eye disease classification task for ResNet-50 with 91.60% accuracy and CutMix yields the greatest improvement for ViT-B with 97.94% accuracy. Code will be available at https://github.com/AIGeeksGroup/MediAug.

<p float="left">
  <img src="https://github.com/AIGeeksGroup/MediAug/blob/main/6_methods.png" width="49%" />
  <img src="https://github.com/AIGeeksGroup/MediAug/blob/main/brain.png" width="49%" />
</p>

## Resource: Visual Augmentation Papers
A comprehensive resource list of visual augmentation is available at [***Data Augmentation in General CV***](https://github.com/AIGeeksGroup/MediAug/blob/main/DA.md).
---

## 🔧 Installation & Setup

```bash
git clone https://github.com/AIGeeksGroup/MediAug.git
cd MediAug
pip install -r requirements.txt
```

To use on **Google Colab** or **Kaggle**, enable GPU and configure data mounting as required.

---

## 📁 Dataset

We use two publicly available medical imaging datasets hosted on Kaggle. In our experiments, the datasets were manually uploaded to Google Drive and accessed through Google Colab notebooks, where all training and evaluation were performed with GPU support.

### 🧿 Eye Diseases Classification (RGB)

* **URL**: [https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
* Classes: Cataract, Diabetic Retinopathy, Glaucoma, Normal
* Balanced dataset
* Random split: 80% train / 20% test

### 🧠 Brain Tumor MRI Classification (Grayscale)

* **URL**: [https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data)
* Classes: Glioma, Meningioma, Pituitary, No Tumor
* Imbalanced dataset
* Random split: 80% train / 20% test

---

## 🏗️ Method Overview

We evaluate six mix-based visual augmentation techniques:

* `MixUp`: Interpolation between image-label pairs
* `YOCO`: Patch-based diverse local/global transforms
* `CropMix`: Multi-scale random crop blending
* `CutMix`: Box-replace image regions + interpolated labels
* `AugMix`: Diverse chained augmentations with consistency
* `SnapMix`: CAM-based semantic-aware mixing

Each method is evaluated on two backbones:

* **ResNet-50** (CNN)
* **ViT-B** (Transformer)

---

## 🧪 Experiments

### ✳️ Comparative Study

| Dataset     | Model     | Best Aug | Accuracy |
| ----------- | --------- | -------- | -------- |
| Brain MRI   | ResNet-50 | MixUp    | 79.19%   |
| Brain MRI   | ViT-B     | SnapMix  | 99.44%   |
| Eye Disease | ResNet-50 | YOCO     | 91.60%   |
| Eye Disease | ViT-B     | CutMix   | 97.94%   |

### 🔬 Ablation Study

Hyperparameter sweep for CutMix (alpha). Best performance at:

* ResNet-50: α = 1.0 → 91.83% Accuracy
* ViT-B: α = 1.0 → 97.94% Accuracy

---

## 💻 Training & Evaluation

To run an experiment with MediAug, follow these steps:

1. **Choose dataset**: `eye` or `brain`
2. **Select model**: `resnet50` or `vit_b`
3. **Pick augmentation method**: one of `mixup`, `cutmix`, `snapmix`, `yoco`, `cropmix`, `augmix`

### Example Commands

Run brain tumor classification with ViT-B and SnapMix:

```bash
python train.py --dataset brain --model vit_b --aug snapmix
```

Run eye disease classification with ResNet-50 and YOCO:

```bash
python train.py --dataset eye --model resnet50 --aug yoco
```

Evaluate a trained model on the test set:

```bash
python evaluate.py --dataset brain --model vit_b --checkpoint ./checkpoints/vit_b_snapmix.pt
```

Visualize augmentation effects (optional):

```bash
python visualize.py --dataset eye --aug mixup --output_dir ./visuals
```

Training details:

* Epochs: 50
* Optimizer: Adam
* Learning Rate: 0.001
* Batch Size: 32
* Image Size: 224×224
* GPU: Tesla T4 or A100 (Google Colab, via mounted Google Drive)
* CPU: Intel Xeon, 80GB RAM

> **Note:** All experiments were conducted on Google Colab. The datasets were uploaded to Google Drive and accessed using standard Colab notebook mounts (e.g., `from google.colab import drive`). Kaggle was not used for runtime.

* Epochs: 50
* Optimizer: Adam
* Learning Rate: 0.001
* Image Size: 224x224
* Hardware: Tesla T4 / A100, Intel Xeon CPU, 80GB RAM

```bash
python train.py --dataset eye --model resnet50 --aug mixup
```

---

## 📈 Visualization

Side-by-side augmentation previews:

---

## 🧠 Model Zoo

| Model     | Dataset | Aug     | Accuracy |
| --------- | ------- | ------- | -------- |
| ResNet-50 | Eye     | YOCO    | 91.60%   |
| ViT-B     | Brain   | SnapMix | 99.44%   |


---

## 📜 License & Acknowledgements

Released under MIT License. Thanks to:

* Authors of MixUp, CutMix, SnapMix
* Datasets from Kaggle
* Pretrained MedConv and JointViT models
* MIUA 2025 for support

For questions, contact [**y.zhao2@latrobe.edu.au**](mailto:y.zhao2@latrobe.edu.au).


