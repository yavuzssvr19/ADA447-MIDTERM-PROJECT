# Automating Pneumonia Diagnosis: A Deep Learning Journey with Fastai

## Overview

This project implements a comprehensive deep learning pipeline for **pneumonia detection** from chest X-ray images using the **Fastai** library. It follows the structured steps outlined including data preparation, DataBlock creation, model training, evaluation, and advanced techniques like learning rate finder and transfer learning.

## Project Structure

```
├── ada447-midterm-project (1).ipynb   # Main Jupyter Notebook
├── IM-0001-0001.jpeg                  # Example pneumonia image
├── test_normal.jpeg                   # Example normal image
└── README.md                          # Project documentation
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yavuzssvr19/ADA447-MIDTERM-PROJECT.git
   cd ADA447-MIDTERM-PROJECT
   ```
2. **Set up the environment** (using Conda):
   ```bash
   conda create -n ada447 python=3.11
   conda activate ada447
   pip install fastai pandas matplotlib kaggle
   ```

## Data Acquisition

### A.1 Download the Data

The Chest X-Ray Pneumonia dataset is obtained from Kaggle:

```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia --unzip -p data
```

### A.1.1 Inspect the Data Layout

```
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/  (if available)
```

### A.1.2 DataBlock Planning

We leverage the folder structure for labeling. `train` and `test` folders are used by Fastai's `GrandparentSplitter` to split the data.

## DataBlock & DataLoaders

### A.2.1 Define Blocks

```python
from fastai.vision.all import *

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=GrandparentSplitter(train_name='train', valid_name='test'),
    get_y=parent_label,
    item_tfms=Resize(460),
    batch_tfms=aug_transforms(size=224)
)
```

### A.2.2 Fetch Items into DataBlock

```python
path = Path('data/chest_xray')
dls = dblock.dataloaders(path)
```

### A.2.3 Define Labels

Labels are inferred from parent directory names (`NORMAL` or `PNEUMONIA`) via `parent_label`.

### A.2.4 Data Transformations

- **Item transforms**: `Resize(460)` scales images to 460×460 to speed up augmentation.
- **Batch transforms**: `aug_transforms(size=224)` applies flips, rotations, zooms, and lighting changes to 224×224 crops.

## A Word on Presizing

Fastai recommends resizing **before** augmentations to avoid artifacts:

1. Increase the size (item by item)
2. Apply augmentation (batch by batch)
3. Decrease the size (batch by batch)

## Inspecting the DataBlock

### A.3.1 Show a Batch

```python
dls.show_batch(max_n=9, figsize=(6,6))
```

### A.3.2 Check Labels

Visual check ensures class balance and correct labeling.

### A.3.3 Summarize the DataBlock

```python
dls.summary()
```

Provides counts per class, batch shape, and transformations applied.

## Model Training

### A.4.1 Baseline Model

A simple CNN benchmark using a pretrained `resnet34`:

```python
learn = cnn_learner(dls, resnet34, metrics=accuracy)
learn.fine_tune(3)
```

### A.4.2 Model Interpretation

Inspect predictions and top losses:

```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9)
```

### A.4.3 Confusion Matrix

```python
interp.plot_confusion_matrix(figsize=(4,4))
```

## Advanced Techniques

### B.1 Learning Rate Finder

```python
learn.lr_find()
```

Helps select an optimal learning rate between too small (slow convergence) and too large (divergence).

### B.2 Finder Algorithm

- Start with very low LR
- Train one batch, record loss
- Double LR and repeat until loss increases sharply

### B.3 Transfer Learning

- **Freezing**: Train only the final layer
- **Unfreezing**: Fine-tune all layers

```python
learn.freeze()
learn.fine_tune(5)
```

### B.4 Discriminative Learning Rates

Use a range of LRs for different layers:

```python
learn.unfreeze()
learn.fit_one_cycle(5, lr_max=slice(1e-6,1e-4))
```

### B.5 Epoch Selection

Decide epochs based on LR finder results and avoid early stopping conflicts.

### B.6 Model Capacity & Mixed Precision

- Increase model size (e.g., `resnet50`) with smaller batch size
- Use mixed precision:

```python
learn.to_fp16()
```

## Results

- Baseline accuracy: **XX%**
- Final accuracy after fine-tuning: **YY%**
- Confusion matrix and loss curves are available in the notebook.

## Running the Notebook

Open `ada447-midterm-project (1).ipynb` and execute all cells in order:

```bash
jupyter notebook "ada447-midterm-project (1).ipynb"
```

## Dependencies

- Python 3.11
- fastai 2.x
- PyTorch
- pandas
- matplotlib
- Kaggle API


