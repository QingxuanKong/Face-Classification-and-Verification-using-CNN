# Face Classification & Verification using CNN

This project implements a solution for the [11785 HW2P2 Face Verification Spring 2025](https://www.kaggle.com/competitions/11785-hw-2-p-2-face-verification-spring-2025). The solution ranked 22 / 301, achieving an EER of 0.02747 on the test dataset.

## Project Overview

The project address two related but distinct tasks: face classification and face verification.

- Face Classifier: trains a model to identify the person in the image from a set of known identities by extracting discriminative feature vectors from face images.

- Face Verification: determines whether two faces belong to the same person by measuring the similarity between feature vectors, even if the person was not in the training dataset.

## Project Highlights

- Leveraged embeddings learned during face classification to perform face verification.
- Implemented an enhanced MobileFaceNet architecture with channel attention mechanism for efficient face recognition.
- Enhanced the feature separability through a hybrid loss function combining cross-entropy and Arcface loss.

## Dataset

This project utilizes two complementary datasets: classification dataset and verification dataset.

- Classification Dataset: is a subset of the VGGFace2 dataset for training and validation. It consists of 8,631 identities with image resolution of 112 x 112.

- Verification Dataset: is used for validation and testing. It consists of 6,000 image pairs with 5,749 identities in total.

To enrich the dataset and improve the model's generalization, several data augmentation are applied:

```bash
transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.2))
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomRotation(degrees=15)
transforms.v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
```

## Model Architecture

The projects employs an enhanced version of MobileFaceNet for face recognition tasks for its accuracy and efficiency. It is a lightweight archiecture for face recognition on resource-constrained devices. Key architectural components include:

- Bottleneck Layers: reduces computation while preserving essential spatial features. Each bottleneck includes depthwise and pointwise convolution. Depthwise convolution extracts spatial features independently for each channel Pointwise convolution mixes information across channels and adjusts the number of output channels.

- Feature Aggregation: replaces traditional global average pooling with 1 x 1 convolutions. It learns channel-wise relationships more efficiently than uniform pooling.

- Attention Mechanism: Channel Attention added in the third bottleneck layers emphasizes important feature channels. CBAM (Convolutional Block Attention Module) added in the fifth bottleneck layers combines spatial and channel attention to highlight important spatial regions and feature channels.

## Loss Function

The challenge in face recognition is optimizing feature space to maximize intra-class similarity while maximizing inter-class separation.

The verification objective can be expressed as $L = \max(s^n - s^p + m, 0)$, where

- $s^p$ represents the similarity score between the anchor feature $z^i$ and a positive feature $z^p_j$ from the same identity.
- $s^n$ represents the similarity score between the anchor feature $z^i$ and a negative feature $z^n_k$ from a different identity.
- $m$ is a margin parameter that enforces a gap between the positive and negative similarity scores.

The objective of this loss function is to ensure that $s^n$ is at least $m$ units less than $s^p$.

The project implements a hybrid loss function combining:

- Cross-Entropy Loss (weight 1.0): is a standard classification loss that encourages correct identity predictions, but it does not explicitly enforce margin between classes

- ArcFace Loss (weight 1.5): improves feature discrimination by adding an angular margin between feature vectors. It Transforms the standard softmax function $s_j = z_i \cdot w_j = \|z_i\| \|w_j\| \cos \theta_j$ (where $θ_j$ is the angle between the feature vector $z_i$ and the weight vector $w_j$) by introducing an angular margin $m$ to $\theta_j$.

## Evaluation Metric

Face Classification is evaluated with accuracy:

- Accuracy = # of correctly classified images / total images

Face Verification is evaluated using the Equal Error Rate (EER). The EER is the point at which the rate of false acceptances (False Acceptance Rate, FAR) equals the rate of false rejections (False Rejection Rate, FRR). These rates are defined as follows:

- FAR = # of false acceptances / total # of impostor attempts
- FRR = # of false rejections / total # of genuine attempts

The EER is the value where these two rates are equal, indicating the threshold at which
the system’s error rates are balanced. A lower EER indicates better performance.

- EER = FAR = FRR

## Model Performance

The model achieved excellent performance across evaluation metrics:

- Accuracy on validation dataset: 98.3%
- EER on validation dataset: 0.02240
- EER on test dataset: 0.02747

## Installation

- Download the dataset

  ```bash
  !mkdir 'content/data'
  !kaggle competitions download -c 11785-hw-2-p-2-face-verification-spring-2025
  !unzip -qo '11785-hw-2-p-2-face-verification-spring-2025' -d 'content/data'
  ```

- Adjust Configuration  
  Modify the num_workers nased on available CPU cores. Set pin_memory=True for GPU acceleration.

- Run the Script  
  Execute the jupyter notebook to train and evaludate the model.

## Experiment Tracking

All the alations are documented in https://wandb.ai/islakong-carnegie-mellon-university/hw2p2/workspace?nw=nwuserislakong

The best result is in https://wandb.ai/islakong-carnegie-mellon-university/hw2p2/runs/8nkdrhce?nw=nwuserislakong
