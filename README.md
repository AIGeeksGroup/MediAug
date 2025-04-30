# MediAug
This is the code repository for the paper:
> **MediAug: Exploring Visual Augmentation in Medical Imaging**
>
> Xuyin Qi\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*<sup>†</sup>, Canxuan Gang\*, Hao Zhang, Lei Zhang, Zhiwei Zhang, and [Yang Zhao](https://yangyangkiki.github.io/)<sup>#</sup>
>
> \*Equal contribution. <sup>†</sup>Project lead. <sup>#</sup>Corresponding author.
>
>
> **[[arXiv]](https://arxiv.org/abs/2504.09518)** **[[Paper with Code]](https://paperswithcode.com/paper/3d-coca-contrastive-learners-are-3d)** **[[HF Paper]](https://huggingface.co/papers/2504.09518)**

![image](https://github.com/AIGeeksGroup/MediAug/blob/main/arch.png)


## Citation

If you use any content of this repo for your work, please cite the following our paper:
```

```

## Introduction
Data augmentation is essential in medical imaging for improving classification accuracy, lesion detection, and organ segmentation under limited data conditions. However, two significant challenges remain. First, a pronounced domain gap between natural photographs and medical images can distort critical disease features. Second, augmentation studies in medical imaging are fragmented and limited to single tasks or architectures, leaving the benefits of advanced mix-based strategies unclear. To address these challenges, we propose a unified evaluation framework with six mix-based augmentation methods integrated with both convolutional and transformer backbones on brain tumour MRI and eye disease fundus datasets. Our contributions are threefold. (1) We introduce MediAug, a comprehensive and reproducible benchmark for advanced data augmentation in medical imaging. (2) We systematically evaluate MixUp, YOCO, CropMix, CutMix, AugMix, and SnapMix with ResNet-50 and ViT-B backbones. (3) We demonstrate through extensive experiments that MixUp yields the greatest improvement on the brain tumor classification task for ResNet-50 with 79.19% accuracy and SnapMix yields the greatest improvement for ViT-B with 99.44% accuracy, and that YOCO yields the greatest improvement on the eye disease classification task for ResNet-50 with 91.60% accuracy and CutMix yields the greatest improvement for ViT-B with 97.94% accuracy. Code will be available at https://github.com/AIGeeksGroup/MediAug.

<p float="left">
  <img src="https://github.com/AIGeeksGroup/MediAug/blob/main/6_methods.png" width="49%" />
  <img src="https://github.com/AIGeeksGroup/MediAug/blob/main/brain.png" width="49%" />
</p>
