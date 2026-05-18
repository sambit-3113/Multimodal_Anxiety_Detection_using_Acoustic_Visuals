# Transfer Learning Module

This folder contains different **Transfer Learning (TL)** strategies implemented for multimodal transformer-based architectures.

The main implementation is located in:

```bash
utils/transfer.py
```

The module provides multiple transfer learning approaches with different freezing and fine-tuning strategies for pretrained models.

---

# Features

- Load pretrained checkpoints
- Freeze complete backbone
- Train only classifier layers
- Partial transformer fine-tuning
- Multimodal-only fine-tuning
- Configurable freezing depth
- Modular TL strategies

---

# File Structure

```bash
utils/
└── transfer.py

mainkfold.py
config.yaml
```

---

# Transfer Learning Modes

Transfer learning behavior is controlled using:

```python
args.tl_mode
```

Currently implemented modes:

---

## 1. FC Only Fine-Tuning (`tl_mode="fc"`)

Only the classifier layers are trainable.

### Behavior

- Entire backbone frozen
- Only `fc` / `output` layers updated

### Use Case

Useful when:
- Dataset is small
- Preventing overfitting
- Fast experimentation

---

## 2. Partial Fine-Tuning of All Transformers (`tl_mode="all"`)

Partially fine-tunes:
- Audio encoder
- Visual encoder
- Multimodal encoder

### Behavior

- Early transformer layers frozen
- Later layers trainable
- Classifier remains trainable

### Controlled By

```python
args.num_freeze
```

Example:

```python
num_freeze = 2
```

This freezes the first 2 layers of each transformer encoder.

---

## 3. Multimodal-Only Fine-Tuning (`tl_mode="multimodal"`)

Fine-tunes only the multimodal encoder.

### Behavior

- Audio encoder frozen
- Visual encoder frozen
- Only later layers of multimodal encoder trained

### Use Case

Useful when:
- Audio/visual features are already well learned
- Only fusion adaptation is needed

---

# Important Functions in `transfer.py`

## Load Pretrained Weights

```python
load_pretrained(net, path, device)
```

Loads pretrained checkpoint into the model.

---

## Freeze Entire Network

```python
freeze_all(net)
```

Disables gradient updates for all parameters.

---

## Unfreeze Classifier

```python
unfreeze_classifier(net)
```

Enables training only for:
- `fc`
- `output`

layers.

---

## Partial Transformer Freezing

### All Encoders

```python
freeze_partial_all_transformers(net, num_freeze)
```

Freezes the first `N` layers of:
- audio encoder
- visual encoder
- multimodal encoder

---

### Multimodal Only

```python
freeze_partial_multimodal(net, num_freeze)
```

Freezes:
- audio encoder
- visual encoder

Partially trains:
- multimodal encoder

---

# Commented Experimental Blocks

Inside `utils/transfer.py`, several older or experimental transfer learning approaches are commented out.

These blocks can be reused when needed for:
- ablation studies
- experimentation
- alternate freezing policies
- comparison studies

You can simply uncomment and modify the required section.

Examples include:
- classifier-only training
- transformer partial freezing
- multimodal selective fine-tuning

---

# How to Use

Transfer learning settings are controlled through:

```yaml
config.yaml
```

Example configuration:

```yaml
pretrained_path: "weights/model.pt"

freeze_backbone: true

tl_mode: "multimodal"

num_freeze: 2
```

---

# Running K-Fold Training

The repository also contains:

```bash
mainkfold.py
```

This script performs:

- K-Fold Cross Validation
- Training
- Validation
- Model checkpoint saving
- WandB logging
- Metric computation

---

# Supported Models

`mainkfold.py` currently supports:

- `TMeanNet`
- `DepressionDetector`
- `TAMFN`
- `ChunkCrossAttentionNet`
- `ChunkTransformerNet`

---

# Metrics Computed

During validation, the following metrics are computed:

- Accuracy
- Precision
- Recall
- F1 Score
- Balanced Accuracy
- MCC
- Weighted Precision
- Weighted Recall
- Weighted F1

Confusion matrix statistics are also printed.

---

# Model Saving Structure

## Standard Training

A `weights/` folder is created automatically.

Saved checkpoints follow:

```bash
weights/
└── model_dataset.pt
```

Example:

```bash
weights/
└── TAMFN_dvlog.pt
```

---

## K-Fold Training

A `weights_kfold/` folder is created automatically.

Structure:

```bash
weights_kfold/
└── model_dataset/
    ├── fold1.pt
    ├── fold2.pt
    ├── fold3.pt
    ├── fold4.pt
    └── fold5.pt
```

Example:

```bash
weights_kfold/
└── TAMFN_dvlog/
    ├── fold1.pt
    ├── fold2.pt
    ├── fold3.pt
    ├── fold4.pt
    └── fold5.pt
```

Each fold stores the best-performing checkpoint for that fold.

---

# WandB Integration

During training, a `wandb/` folder is automatically created.

It stores:
- experiment runs
- logs
- metrics
- visualizations
- model artifacts

Best models for each fold are also logged as WandB artifacts for experiment tracking and visualization.

---

# Example Run

```bash
python mainkfold.py
```

or

```bash
python mainkfold.py \
    -m TAMFN \
    --num_folds 5
```

---

# Notes

- Transfer learning is optional.
- If `pretrained_path=None`, training starts from scratch.
- The TL pipeline is fully configurable through `config.yaml`.
- Additional transfer learning strategies can be easily added to `utils/transfer.py`.

---
