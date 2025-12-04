# Autism Detection Using YOLOv8 Classification (SGD Optimizer)

This project implements an **autism vs non-autism image classifier** using **YOLOv8m-cls** from the Ultralytics library.  
The model is trained with **SGD optimizer**, strong data augmentation, and evaluated using a separate test set.

---

## Project Overview

The goal of this project is to classify facial images into:

- **Autistic**
- **Non-Autistic**

This is done using YOLOv8 *classification* (not object detection).  
The dataset is organized in ImageNet-style directory format:
```

dataset/
│
├── train/
│ ├── autistic/
│ └── non_autistic/
│
├── valid/
│ ├── autistic/
│ └── non_autistic/
│
└── test/
├── autistic/
└── non_autistic/
```

The model is trained for **400 epochs** with SGD, using augmentations such as flips, scaling, and rotations.

---

## Features

- Uses **YOLOv8m-cls** pre-trained weights
- SGD optimizer for stable classification training
- Strong augmentation for better generalization
- Autotuned TensorFlow pipeline (used only for dataset preparation)
- Full evaluation on test dataset (accuracy, precision, recall, F1)
- Supports GPU if available

---

## Model Training Code

Key training configuration:

```python
model = YOLO("yolov8m-cls.pt")

results = model.train(
    data="./dataset",
    epochs=400,
    imgsz=224,
    batch=64,
    optimizer="SGD",
    lr0=0.001,
    momentum=0.97,
    weight_decay=0.0005,
    augment=True,
    flipud=0.5,
    fliplr=0.5,
    scale=0.5,
    degrees=15,
    patience=50,
    project="autism_yolov8_sgd",
    name="yolov8m_sgd_400e"
)
```
## After training, the best model is loaded and evaluated on the test set
```python
model = YOLO("./runs/classify/train/weightsYOLOv8/best.pt")
metrics = model.val(data="./dataset/test")
print(metrics)
```
## Metrics include:

- **Accuracy**
- **Precision / Recall**
- **F1 Score**
- **Confusion Matrix**

## Installation

git clone https://github.com/ahd01/autism-detection-using-YOLOv8-model.git


cd autism-detection-using-YOLOv8-model

## Install all requirements
pip install -r requirements.txt
