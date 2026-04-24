# 🧩 Autism Emotion Detection with VGG16 Transfer Learning

> **A VGG16-based CNN trained on Dr. Fatma M. Talaat's Autistic Children Emotions dataset to classify six emotional expressions, supporting research into assistive technology for autism spectrum disorder (ASD).**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00)](https://www.tensorflow.org/)
[![VGG16](https://img.shields.io/badge/Backbone-VGG16%20ImageNet-8B5CF6)](https://keras.io/api/applications/vgg/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Why This Matters](#why-this-matters)
- [The Six Emotions](#the-six-emotions)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Step 1: Environment Setup](#step-1-environment-setup)
  - [Step 2: Image Size and Data Augmentation](#step-2-image-size-and-data-augmentation)
  - [Step 3: Data Loading](#step-3-data-loading)
  - [Step 4: Model Architecture](#step-4-model-architecture)
  - [Step 5: Training](#step-5-training)
  - [Step 6: Evaluation](#step-6-evaluation)
  - [Step 7: Predictions](#step-7-predictions)
- [Key Design Decisions](#key-design-decisions)
- [Results](#results)
- [How to Reproduce](#how-to-reproduce)
- [Ethical Considerations](#ethical-considerations)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [Roadmap](#roadmap)
- [Dataset Acknowledgement](#dataset-acknowledgement)
- [Author](#author)
- [License](#license)

---

## Overview

This project trains a **transfer-learning CNN** to classify emotional expressions in photographs of children on the autism spectrum. The backbone is **VGG16** pretrained on ImageNet, with its convolutional layers frozen and a custom classifier head trained on top to output one of six emotion classes.

The entire pipeline runs in a single Kaggle notebook, from data loading through augmentation, training, evaluation, and qualitative inference on held-out test images.

The technical core is deliberately simple. The ambition is not to invent a new architecture, it is to produce a reliable baseline on a specialised, under-studied dataset that could plausibly support downstream assistive technology research.

---

## Why This Matters

Children on the autism spectrum often experience the world of facial expressions differently from neurotypical children. Some have difficulty producing the facial expressions associated with specific emotions. Some have difficulty reading them in others. This creates a real, measurable gap in how existing emotion recognition systems (trained overwhelmingly on neurotypical faces like FER2013) perform when deployed with autistic users.

A model trained specifically on facial expressions of autistic children addresses two gaps at once:

1. **Representation.** Existing emotion recognition datasets rarely include autistic children. A model trained on FER2013 or AffectNet will predict poorly on the population it is supposed to serve in an assistive context
2. **Application.** There are genuine, ethical assistive-technology use cases for this kind of model — emotion-aware educational tools, therapist-in-the-loop communication aids, research into expression patterns across the spectrum

This project is an early-stage research baseline. It is not a deployed product. The [Ethical Considerations](#ethical-considerations) section later in this README is required reading before anyone considers extending this work toward real-world deployment.

---

## The Six Emotions

The model predicts one of six emotion classes, derived from the dataset's native labels:

| Label | Emotion |
|-------|---------|
| 0 | **Surprise** |
| 1 | **Delight** |
| 2 | **Sadness** |
| 3 | **Fear** |
| 4 | **Joy** |
| 5 | **Anger** |

Worth noting: this six-class scheme diverges from the standard **Ekman-seven** (Happy, Sad, Angry, Fear, Disgust, Surprise, Neutral) that drives most facial expression datasets including FER2013. Two meaningful differences:

- **No "Disgust" class.** Disgust is the most consistently hardest-to-detect emotion in FER datasets even with large sample sizes. Its omission here is a reasonable curatorial choice
- **"Delight" and "Joy" as separate classes.** Most datasets collapse these into a single "Happy" class. Splitting them allows the model to distinguish between general positive affect (Joy) and intense momentary positive reactions (Delight). Whether this split is robust in practice is an empirical question the model's per-class performance will answer
- **No "Neutral" class.** Neutral faces are not represented, meaning the model will always predict one of the six active emotions. If deployed, this is a constraint that matters

---

## Dataset

**Name:** Autistic Children Emotions Dataset
**Curator:** Dr. Fatma M. Talaat
**Kaggle path:** `/kaggle/input/autistic-children-emotions-dr-fatma-m-talaat/`
**Classes:** 6 emotions (Surprise, Delight, Sadness, Fear, Joy, Anger)
**Pre-split into:** `Train/` and `Test/` directories

The dataset is pre-partitioned into training and testing directories, with each emotion as its own subfolder:

```
Autistic Children Emotions - Dr. Fatma M. Talaat/
├── Train/
│   ├── anger/
│   ├── delight/
│   ├── fear/
│   ├── joy/
│   ├── sadness/
│   └── surprise/
└── Test/
    ├── anger/
    ├── delight/
    ├── fear/
    ├── joy/
    ├── sadness/
    └── surprise/
```

This notebook uses a **three-way split**:

- **Train** (80% of `Train/`) — fed to the model with augmentation
- **Validation** (20% of `Train/`, via `validation_split=0.2`) — used to monitor training
- **Test** (full `Test/` directory) — clean held-out evaluation

Images are RGB photographs of autistic children, resized to **256×256** on load. The dataset preserves colour information (unlike grayscale datasets such as FER2013), which matters for transfer learning from ImageNet-pretrained weights that expect 3-channel input.

---

## Model Architecture

The network uses **VGG16 transfer learning** with a frozen backbone and a trainable custom classifier head.

```
Input (256, 256, 3)
│
├── VGG16 (frozen, ImageNet weights, include_top=False)
│   ├── Block 1: Conv+Conv+MaxPool   → (128, 128, 64)
│   ├── Block 2: Conv+Conv+MaxPool   → (64, 64, 128)
│   ├── Block 3: Conv+Conv+Conv+MaxPool  → (32, 32, 256)
│   ├── Block 4: Conv+Conv+Conv+MaxPool  → (16, 16, 512)
│   └── Block 5: Conv+Conv+Conv+MaxPool  → (8, 8, 512)
│
└── Custom Classifier Head (trainable) ─────────
    Flatten                       → (32,768,)
    Dense(512, ReLU)
    Dropout(0.5)
    Dense(256, ReLU)
    Dropout(0.3)
    Dense(6, Softmax)             → Output (6 classes)
```

**Why transfer learning?**

A full VGG16 has ~138M parameters. Training from scratch on a small, specialised dataset like this one would almost certainly overfit, and the network would fail to learn useful low-level visual features (edges, textures, face parts) from the limited images available.

Instead, the ImageNet-pretrained convolutional backbone brings **pre-learned visual representations** — the same edge detectors, texture filters, and part detectors that let VGG16 recognise a thousand object categories. These features are transferable to facial expression classification almost for free. Only the classifier head is retrained from scratch to map those features to the six emotion classes.

**Why freeze the backbone?**

With a small dataset, fine-tuning the full 138M-parameter backbone risks catastrophic forgetting and severe overfitting. Freezing it means:

- Training is fast (only ~17M trainable parameters in the head)
- The pretrained visual features are preserved
- The model can't overfit at the feature-extraction layer, only at the classifier

A natural next iteration (see [Roadmap](#roadmap)) would be a two-stage training schedule: train the head first with the backbone frozen, then unfreeze the top few VGG blocks and fine-tune with a very low learning rate.

**Classifier head design:**

- **Flatten layer** converts the (8×8×512) feature maps to a 32,768-dim vector
- **Dense(512) → Dropout(0.5)** provides high capacity with heavy regularisation at the first projection
- **Dense(256) → Dropout(0.3)** refines the representation with lighter dropout
- **Dense(6, Softmax)** outputs a probability distribution over the six classes

---

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Input shape** | 256 × 256 × 3 (RGB) | Matches VGG16's expected input channels, preserves detail |
| **Batch size** | 64 | Standard for mid-size CNNs on T4 GPU |
| **Epochs** | 28 | Tuned empirically for convergence without overfitting |
| **Optimiser** | Adam (lr=0.0001) | Lower than default (1e-3) because only the head is training — a smaller step works better when the backbone is frozen |
| **Loss** | categorical_crossentropy | Standard for multi-class one-hot targets |
| **Validation split** | 0.2 (20% of `Train/`) | Standard holdout for callbacks and monitoring |

### Data Augmentation

Applied to the training set only. Validation and test streams are clean (rescale-only).

| Augmentation | Value | Purpose |
|--------------|-------|---------|
| `rescale` | `1./255` | Normalise pixel values to [0, 1] |
| `rotation_range` | 15° | Simulate camera tilt and natural head movement |
| `width_shift_range` | 0.09 | Small horizontal translation |
| `height_shift_range` | 0.09 | Small vertical translation |
| `shear_range` | 0.09 | Mild affine shearing |
| `zoom_range` | 0.09 | Small zoom in/out |
| `horizontal_flip` | True | Faces are roughly symmetric — doubles effective data |
| `brightness_range` | [0.8, 1.2] | Simulate lighting variation |
| `fill_mode` | nearest | Fills empty pixels after rotation/shift with the nearest pixel value |

The augmentation is deliberately **gentle** (note the 0.09 values for shifts, shear, and zoom). Too-aggressive augmentation on a small specialised dataset can destroy the very facial structure the model needs to read. The values chosen simulate natural variation without distorting emotional features.

---

## Pipeline Walkthrough

The notebook runs seven clearly numbered steps. Below is a detailed walkthrough of each.

---

### Step 1: Environment Setup

**What it does:** Imports TensorFlow, Keras layers (Conv2D, MaxPooling2D, Flatten, Dense, Dropout), the VGG16 application, Adam optimiser, NumPy, Matplotlib, and the ImageDataGenerator. Defines the two dataset paths (`train_data_dir` and `test_data_dir`).

---

### Step 2: Image Size and Data Augmentation

**What it does:** Sets `img_size = (256, 256)` and `batch_size = 64`, then constructs the training `ImageDataGenerator` with all augmentation parameters plus `validation_split=0.2`.

This cell is where every augmentation choice from the table above is applied. It is the single biggest lever for the model's generalisation performance on a small dataset.

---

### Step 3: Data Loading

**What it does:** Creates three `flow_from_directory` generators:

- **Train generator** — `Train/` with `subset='training'`, augmentation active
- **Validation generator** — `Train/` with `subset='validation'`, same augmentation generator (but the split is clean)
- **Test generator** — `Test/` with a separate rescale-only `ImageDataGenerator` (no augmentation, preserving ground-truth images for evaluation)

All three generators use `class_mode='categorical'`, which one-hot encodes the six emotion labels for cross-entropy training.

---

### Step 4: Model Architecture

**What it does:** Loads VGG16 with ImageNet weights and `include_top=False` (discarding the original 1,000-class classifier), freezes every convolutional layer (`layer.trainable = False`), and builds a Sequential model that stacks the frozen VGG16 backbone with the custom classifier head described in the [Architecture section](#model-architecture).

The final output layer uses `len(train_generator.class_indices)` to auto-derive the number of classes from the data directory, which makes the code dataset-agnostic — if the dataset ever changes to 7 or 5 classes, this line still works.

---

### Step 5: Training

**What it does:** Compiles the model with `Adam(learning_rate=0.0001)`, categorical crossentropy loss, and accuracy metric. Runs `model.fit()` for 28 epochs on the train generator with the validation generator as the eval stream.

After training, two plots are produced:

1. **Accuracy curves** — training and validation accuracy over epochs
2. **Loss curves** — training and validation loss over epochs

A `model.summary()` is also printed, showing the frozen VGG16 parameter count (non-trainable), the trainable head parameter count, and the total.

---

### Step 6: Evaluation

**What it does:** Calls `model.evaluate(test_generator)` on the held-out test set, prints the test accuracy and test loss, and visualises the test accuracy as a single bar chart.

This is the headline number that represents the model's generalisation performance on genuinely unseen data.

---

### Step 7: Predictions

**What it does:** Performs two kinds of qualitative inference.

**Single-image prediction:** Loads a specific test image (`Test/fear/13.jpg`), preprocesses it (resize to 256×256, normalise to [0, 1], add batch dimension), runs `model.predict()`, and prints the top predicted emotion using the hard-coded emotion label list: `['Surprise', 'Delight', 'Sadness', 'Fear', 'Joy', 'Anger']`.

**Random-grid prediction:** Samples 20 random images from the test set, predicts each one, and displays a 4×5 grid showing each image with its predicted label and confidence score. This visual qualitative check complements the aggregate test accuracy by showing exactly where the model succeeds and where it fails.

---

## Key Design Decisions

A handful of deliberate choices shape this baseline.

| Decision | Choice | Why |
|----------|--------|-----|
| **Backbone** | VGG16 with ImageNet weights | Transfers low-level visual features for free. Training from scratch on a small specialised dataset would severely overfit |
| **Freezing strategy** | Freeze the entire backbone, train only the head | Safest option for a small dataset. Fine-tuning the backbone needs careful learning rate schedules that are out of scope for a first baseline |
| **Input size** | 256 × 256 RGB | Matches VGG16's natural input scale and keeps full colour information. Using grayscale (like FER2013) would waste the pretrained colour-sensitive filters |
| **Learning rate** | 0.0001 | Ten times smaller than the Adam default. Appropriate when only the classifier head is trainable — large steps on a small parameter set cause oscillation |
| **Augmentation intensity** | Gentle (0.09 for shifts, shear, zoom) | Aggressive augmentation on a small specialised dataset risks distorting emotional features. Gentle augmentation still provides regularisation without destroying signal |
| **Dropout schedule** | 0.5 after Dense(512), 0.3 after Dense(256) | Heavier regularisation at the first projection where overfitting risk is highest. Lighter at the second projection where the feature dimensionality is already reduced |
| **Number of classes** | 6 (native dataset labels) | Preserved as-is from the dataset curator's schema rather than mapping to Ekman-7. The curator's choice reflects domain expertise |
| **Output layer sizing** | Derived from `train_generator.class_indices` | Dataset-agnostic — works if the dataset later adds or removes classes |

---

## Results

> *Fill this section with your actual numbers after running the notebook. Placeholders below show the expected format.*

### Test Set Performance

| Metric | Value |
|--------|-------|
| **Overall Test Accuracy** | *(run notebook)* |
| **Test Loss** | *(run notebook)* |

### Per-Class Observations

After running, inspect:

- Which emotions are most reliably predicted (typically Joy and Anger in facial expression tasks)
- Which emotions are most commonly confused (Fear ↔ Surprise is a known difficult pair)
- Whether the Delight vs Joy distinction — unique to this dataset — holds up, or whether the model collapses them

### Context: Comparable Benchmarks

For reference, published results on autistic-children emotion datasets typically range from:

- **Simple CNN from scratch:** 50–65% (limited by dataset size)
- **Transfer learning (VGG / ResNet / EfficientNet):** 65–80% depending on dataset size and augmentation
- **Fine-tuned transfer learning with two-stage training:** 75–85% on the better-curated datasets

Direct comparison to FER2013 numbers is not meaningful because FER2013 has 35K+ images and uses the Ekman-seven scheme, while this dataset is smaller and uses a different six-class schema.

---

## How to Reproduce

### Option 1: Run on Kaggle (Recommended)

1. Open [Kaggle](https://www.kaggle.com/) and sign in
2. Create a new notebook
3. Attach the dataset from the Kaggle data tab: search for `autistic-children-emotions-dr-fatma-m-talaat` and click **Add**
4. Upload `autism-emotion-detection.ipynb` or copy the code cells
5. Enable **GPU T4 x1** in notebook settings (free tier)
6. Run all cells top to bottom

Expected runtime on T4 GPU: approximately 20–40 minutes.

### Option 2: Run Locally

```bash
# Clone the repo
git clone https://github.com/[your-username]/autism-emotion-detection.git
cd autism-emotion-detection

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle
kaggle datasets download -d <owner>/autistic-children-emotions-dr-fatma-m-talaat
unzip autistic-children-emotions.zip -d data/

# Update paths in the notebook from /kaggle/input/... to data/...
jupyter notebook autism-emotion-detection.ipynb
```

### Hardware Recommendations

- **Minimum:** CPU-only training is impractical (4–8 hours). Not recommended
- **Recommended:** Single modern GPU (T4, RTX 3060+, A10G). 20–40 minute training
- **Best:** A100 or L4. 10–15 minute training

---

## Ethical Considerations

This section is the most important part of this README. Emotion recognition on a clinical population requires careful framing.

### What This Project Is Not

- **Not a medical device.** This is research code. It is not validated, certified, or designed for clinical use
- **Not a diagnostic tool.** This model classifies facial expressions in images. It does not diagnose anything. Autism is not diagnosable from a photograph, and emotion is not inferable from a facial expression alone
- **Not a deployment-ready product.** Any deployment to real users, especially children with ASD, would require substantial additional work on fairness auditing, user consent frameworks, clinical oversight, and bias evaluation

### Dataset Limitations

- **Small sample size.** Specialised datasets like this one are necessarily small. Generalisation beyond the exact data distribution is uncertain
- **Demographic representation.** The dataset was curated in a specific context. Performance on children outside that demographic (different regions, age groups, or severity levels on the spectrum) is unknown
- **Label validity.** "Emotion" labels on facial expressions are always inferential. A photograph labelled "fear" captures a facial expression that a human annotator interpreted as fearful. The actual emotional state of the child at the moment of capture is unknowable
- **Consent and privacy.** Images of children on the autism spectrum are sensitive. Any derivative work should honour the original consent framework under which the dataset was collected

### Considerations for Downstream Use

If anyone were to extend this work toward actual assistive technology, these would be non-negotiable requirements:

- **Clinical collaboration.** Work directly with ASD clinicians, educators, and autistic self-advocates from day one, not retrofitted later
- **Autistic community involvement.** Nothing about us without us — the autistic community must be substantively involved in the design of any tool that uses this technology
- **Uncertainty communication.** The model is a probabilistic classifier. Its confidence scores must be surfaced to any downstream user, not hidden behind a confident-sounding label
- **Opt-in consent.** No passive deployment in classrooms, therapy sessions, or surveillance settings without active, informed consent from children (age-appropriately), parents, and care teams
- **Failure-mode design.** The system must fail safely — a wrong emotion prediction in an educational game has low stakes, the same wrong prediction in a clinical decision has potentially high stakes

### Broader Emotion Recognition Caveats

The same caveats that apply to general facial expression recognition apply here with added force due to the clinical population:

- **Emotion inference is not ground truth.** Models predict visual patterns, not internal states
- **Cultural expression norms vary.** A model trained on one population's expression conventions may misread another's
- **Autistic expressions may differ from neurotypical norms.** This is partly why dedicated datasets like this one exist — but the training labels themselves are interpretations, possibly by neurotypical annotators, of autistic children's expressions. That interpretive layer should not be forgotten

Treat this project as **early-stage research**, publishable as a methods contribution, but never as a deployed system without the infrastructure above.

---

## Repository Structure

```
.
├── README.md                              # This file
├── autism-emotion-detection.ipynb         # Complete training and evaluation notebook
├── requirements.txt                       # Python dependencies
├── outputs/                               # (generated)
│   ├── accuracy_curves.png
│   ├── loss_curves.png
│   └── test_predictions_grid.png
└── LICENSE
```

---

## Dependencies

```
tensorflow>=2.15.0
numpy>=1.26.0
matplotlib>=3.8.0
Pillow>=10.0.0
```

Install with:

```bash
pip install -r requirements.txt
```

On Kaggle, everything is pre-installed. No setup required.

---

## Roadmap

Improvements that could move this from research baseline to something more substantive:

- **Two-stage fine-tuning** — train the classifier head first with VGG16 frozen, then unfreeze the top VGG blocks and fine-tune with a very low learning rate (e.g., 1e-5)
- **Confusion matrix and per-class F1 reporting** — currently the notebook reports only test accuracy; adding full classification metrics would match the depth of the FER2013 companion project
- **Modern backbones** — swap VGG16 for ResNet50, EfficientNet-B0, or a vision transformer (ViT) for better feature quality at the same parameter budget
- **Face detection preprocessing** — MTCNN or MediaPipe to crop tightly around the face before feature extraction
- **Test-time augmentation (TTA)** — averaging predictions across augmented views for a small accuracy boost
- **Grad-CAM visualisation** — show which facial regions the model attends to for each prediction, essential for interpretability in a clinical context
- **Cross-dataset evaluation** — test the model on a neurotypical emotion dataset (FER2013, AffectNet) to quantify how much the specialised training matters
- **Class weighting** — if any emotion has materially fewer samples, apply inverse-frequency weighting to the loss
- **Ensemble with the FER2013 model** — combine predictions from this dataset-specific model with a general-purpose FER model for robustness
- **Clinical validation study** — collaboration with ASD clinicians to evaluate whether model predictions align with expert judgment

---

## Dataset Acknowledgement

This project uses the **Autistic Children Emotions dataset curated by Dr. Fatma M. Talaat**. Full credit for dataset collection, curation, and labelling belongs to Dr. Talaat. Anyone using the dataset should cite the curator's original work and comply with the dataset's licence terms as published on Kaggle.

---

## Author

**Collins Lemeke**

This project sits at the intersection of computer vision and accessibility research. It connects to my wider work on efficient, domain-specific deep learning models that address gaps in mainstream ML training data.

For questions, feedback, or feature requests, open a GitHub issue.

---

## License

MIT License. Free to use, modify, and distribute. See [LICENSE](LICENSE) for full terms.

The Autistic Children Emotions dataset has its own licence and terms of use, separate from this code. Please refer to the [original Kaggle dataset page](https://www.kaggle.com/) for dataset licensing details.

---

> *Built with TensorFlow, Keras, and VGG16 transfer learning. Designed as a research baseline, not a deployed product. Read the Ethical Considerations section before extending this work.*
