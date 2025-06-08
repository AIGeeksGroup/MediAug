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

### 🗂️ Dataset Folder Structure

```
📁 dataset/
├── 📁 brain/
│   ├── AugMix_MRI/
│   ├── CropMix_MRI/
│   ├── CutMix_MRI/
│   ├── MixUp_MRI/
│   ├── Original_MRI/
│   ├── SnapMix_MRI/
│   └── YOCO_MRI/
│       ├── Training/
│       │   ├── glioma_tumor/
│       │   ├── meningioma_tumor/
│       │   ├── no_tumor/
│       │   └── pituitary_tumor/
│       └── Testing/
│           ├── glioma_tumor/
│           ├── meningioma_tumor/
│           ├── no_tumor/
│           └── pituitary_tumor/
├── 📁 eye/
│   ├── AugMix_contenteye_diseases_dataset/
│   ├── CropMix_contenteye_diseases_dataset/
│   ├── CutMix_contenteye_diseases_dataset/
│   ├── MixUp_contenteye_diseases_dataset/
│   ├── Original_contenteye_diseases_dataset/
│   ├── SnapMix_contenteye_diseases_dataset/
│   └── YOCO_contenteye_diseases_dataset/
│       ├── Training/
│       │   ├── cataract/
│       │   ├── diabetic_retinopathy/
│       │   ├── glaucoma/
│       │   └── normal/
│       └── Testing/
│           ├── cataract/
│           ├── diabetic_retinopathy/
│           ├── glaucoma/
│           └── normal/
```


### 🧿 Eye Diseases Classification (RGB)

* **URL**: [https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)
* Classes: Cataract, Diabetic Retinopathy, Glaucoma, Normal
* Balanced dataset
* Random split: 80% train / 20% test

<p align="left">
  <img src="https://github.com/AIGeeksGroup/MediAug/blob/main/eye_disease.jpg" width="28%" />
  <img src="https://github.com/AIGeeksGroup/MediAug/blob/main/eye_tsne.jpg" width="33%" />
</p>

The left pie chart shows the class distribution across the four categories, demonstrating good class balance. The right t-SNE plot provides a feature-level visualization of the high-dimensional distribution of eye disease samples after dimensionality reduction.

### 🧠 Brain Tumor MRI Classification (Grayscale)

* **URL**: [https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri/data)
* Classes: Glioma, Meningioma, Pituitary, No Tumor
* Imbalanced dataset
* Random split: 80% train / 20% test

<p align="left">
  <img src="https://github.com/AIGeeksGroup/MediAug/blob/main/brain_disease.jpg" width="28%" />
  <img src="https://github.com/AIGeeksGroup/MediAug/blob/main/brain_tsne.jpg" width="33%" />
</p>

The pie chart (left) illustrates the class distribution among four tumor categories. The t-SNE plot (right) visualizes the distribution of brain tumor samples in a two-dimensional space, reflecting their separability and overlap in feature space.


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

## 🧠 Model Zoo

| Model     | Dataset | Aug     | Accuracy |
| --------- | ------- | ------- | -------- |
| ResNet-50 | Eye     | YOCO    | 91.60%   |
| ViT-B     | Brain   | SnapMix | 99.44%   |


---

## 📓 Notebooks

The following notebooks train and evaluate models used in our experiments:

* `resnet50.ipynb`: Trains a ResNet-50 model on the selected dataset with different augmentation strategies. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Training%20model/resnet50.ipynb)
* `VIT-B.ipynb`: Trains a ViT-B (Vision Transformer) model on the selected dataset and compares augmentation effects. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/Training%20model/VIT-B.ipynb)

The following notebooks apply batch augmentation and visualization on the full **Brain MRI** dataset:

* `AugMix_brain.ipynb`: Applies AugMix to the entire brain dataset and visualizes a batch of augmented images. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/AugMix_brain.ipynb)
* `CropMix_brain.ipynb`: Performs CropMix augmentation across the brain dataset with comparative visualization. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/CropMix_brain.ipynb)
* `CutMix_brain.ipynb`: Shows CutMix applied to MRI samples in batch for augmentation analysis. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/CutMix_brain.ipynb)
* `MixUp_brain.ipynb`: Executes MixUp over MRI images and plots combined outputs. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/MixUp_brain.ipynb)
* `SnapMix_brain.ipynb`: Demonstrates CAM-based SnapMix on brain images at dataset level. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/SnapMix_brain.ipynb)
* `YOCO_brain.ipynb`: Applies YOCO to a batch of brain samples and shows spatially mixed results. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/YOCO_brain.ipynb)

The following notebooks apply batch augmentation and visualization on the full **Eye Disease** dataset:

* `AugMix_eye.ipynb`: Applies AugMix on the entire eye disease dataset with visual comparisons. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/AugMix_eye.ipynb)
* `CropMix_eye.ipynb`: Runs CropMix augmentation over eye images and displays batched transformations. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/CropMix_eye.ipynb)
* `CutMix_eye.ipynb`: Demonstrates CutMix applied to eye fundus images with batch-level visualization. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/CutMix_eye.ipynb)
* `MixUp_eye.ipynb`: Mixes image-label pairs from the eye dataset and renders visual effects. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/MixUp_eye.ipynb)
* `SnapMix_eye.ipynb`: Showcases SnapMix on eye disease samples with semantic-preserving augmentation. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/SnapMix_eye.ipynb)
* `YOCO_eye.ipynb`: Uses YOCO to enhance eye data samples with region-wise mixed transforms. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/YOCO_eye.ipynb)

The following notebooks demonstrate how each augmentation method is applied to a single medical image:

* `AugMix_for_single_picture.ipynb`: Applies AugMix transformations step-by-step to one image and visualizes the results. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/AugMix_for_single_picture.ipynb)
* `CropMix_for_single_picture.ipynb`: Demonstrates the CropMix augmentation process with visualization on a single image. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/CropMix_for_single_picture.ipynb)
* `CutMix_for_single_picture.ipynb`: Simulates CutMix augmentation by mixing image patches and overlays on one image. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/CutMix_for_single_picture.ipynb)
* `MixUp_for_single_picture.ipynb`: Shows how MixUp blends two images and labels, visualized clearly. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/MixUp_for_single_picture.ipynb)
* `SnapMix_for_single_picture.ipynb`: Explains SnapMix strategy by combining semantic patches with attention maps. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/SnapMix_for_single_picture.ipynb)
* `YOCO_for_single_picture.ipynb`: Visualizes YOCO's patch-wise mixed local augmentations on a single image. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/YOCO_for_single_picture.ipynb)

The following notebooks apply batch augmentation and visualization on the full **Brain MRI** dataset:

* `AugMix_brain.ipynb`: Applies AugMix to the entire brain dataset and visualizes a batch of augmented images. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/AugMix_brain.ipynb)
* `CropMix_brain.ipynb`: Performs CropMix augmentation across the brain dataset with comparative visualization. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/CropMix_brain.ipynb)
* `CutMix_brain.ipynb`: Shows CutMix applied to MRI samples in batch for augmentation analysis. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/CutMix_brain.ipynb)
* `MixUp_brain.ipynb`: Executes MixUp over MRI images and plots combined outputs. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/MixUp_brain.ipynb)
* `SnapMix_brain.ipynb`: Demonstrates CAM-based SnapMix on brain images at dataset level. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/SnapMix_brain.ipynb)
* `YOCO_brain.ipynb`: Applies YOCO to a batch of brain samples and shows spatially mixed results. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/YOCO_brain.ipynb)

The following notebooks apply batch augmentation and visualization on the full **Eye Disease** dataset:

* `AugMix_eye.ipynb`: Applies AugMix on the entire eye disease dataset with visual comparisons. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/AugMix_eye.ipynb)
* `CropMix_eye.ipynb`: Runs CropMix augmentation over eye images and displays batched transformations. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/CropMix_eye.ipynb)
* `CutMix_eye.ipynb`: Demonstrates CutMix applied to eye fundus images with batch-level visualization. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/CutMix_eye.ipynb)
* `MixUp_eye.ipynb`: Mixes image-label pairs from the eye dataset and renders visual effects. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/MixUp_eye.ipynb)
* `SnapMix_eye.ipynb`: Showcases SnapMix on eye disease samples with semantic-preserving augmentation. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/SnapMix_eye.ipynb)
* `YOCO_eye.ipynb`: Uses YOCO to enhance eye data samples with region-wise mixed transforms. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/YOCO_eye.ipynb)

The following notebooks apply batch augmentation and visualization on the full **Brain MRI** dataset:

* `AugMix_brain.ipynb`: Applies AugMix to the entire brain dataset and visualizes a batch of augmented images. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/AugMix_brain.ipynb)
* `CropMix_brain.ipynb`: Performs CropMix augmentation across the brain dataset with comparative visualization. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/CropMix_brain.ipynb)
* `CutMix_brain.ipynb`: Shows CutMix applied to MRI samples in batch for augmentation analysis. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/CutMix_brain.ipynb)
* `MixUp_brain.ipynb`: Executes MixUp over MRI images and plots combined outputs. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/MixUp_brain.ipynb)
* `SnapMix_brain.ipynb`: Demonstrates CAM-based SnapMix on brain images at dataset level. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/SnapMix_brain.ipynb)
* `YOCO_brain.ipynb`: Applies YOCO to a batch of brain samples and shows spatially mixed results. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/YOCO_brain.ipynb)

The following notebooks demonstrate how each augmentation method is applied to a single medical image:

* `AugMix_for_single_picture.ipynb`: Applies AugMix transformations step-by-step to one image and visualizes the results. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/AugMix_for_single_picture.ipynb)
* `CropMix_for_single_picture.ipynb`: Demonstrates the CropMix augmentation process with visualization on a single image. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/CropMix_for_single_picture.ipynb)
* `CutMix_for_single_picture.ipynb`: Simulates CutMix augmentation by mixing image patches and overlays on one image. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/CutMix_for_single_picture.ipynb)
* `MixUp_for_single_picture.ipynb`: Shows how MixUp blends two images and labels, visualized clearly. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/MixUp_for_single_picture.ipynb)
* `SnapMix_for_single_picture.ipynb`: Explains SnapMix strategy by combining semantic patches with attention maps. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/SnapMix_for_single_picture.ipynb)
* `YOCO_for_single_picture.ipynb`: Visualizes YOCO's patch-wise mixed local augmentations on a single image. [View notebook](https://github.com/AIGeeksGroup/MediAug/blob/main/YOCO_for_single_picture.ipynb)

---

## 📜 License & Acknowledgements

Released under MIT License. Thanks to:

* Authors of MixUp, CutMix, SnapMix
* Datasets from Kaggle
* Pretrained MedConv and JointViT models
* MIUA 2025 for support

For questions, contact [**y.zhao2@latrobe.edu.au**](mailto:y.zhao2@latrobe.edu.au).


