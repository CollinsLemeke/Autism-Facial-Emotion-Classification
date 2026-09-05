# Autism Emotion Detection with VGG16 Transfer Learning

> **A VGG16-based CNN trained on Dr. Fatma M. Talaat's Autistic Children Emotions dataset to classify six emotional expressions, supporting research into assistive technology for autism spectrum disorder (ASD).**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00)](https://www.tensorflow.org/)
[![VGG16](https://img.shields.io/badge/Backbone-VGG16%20ImageNet-8B5CF6)](https://keras.io/api/applications/vgg/)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-70.7%25%20(n%3D75)-yellow)](#results)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF)](https://www.kaggle.com/code/collinslemeke/autism-emotion-detection)
[![Published](https://img.shields.io/badge/Published-IEEE%20IC3ECSBHI%202026-success)](#published-work)
[![License](https://img.shields.io/badge/Code-MIT-lightgrey)](LICENSE)

---

## Table of Contents

- [Overview](#overview)
- [Why This Matters](#why-this-matters)
- [Published Work](#published-work)
- [The Six Emotions](#the-six-emotions)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Key Design Decisions](#key-design-decisions)
- [Results](#results)
- [Read This Before Quoting the Accuracy](#read-this-before-quoting-the-accuracy)
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

**Headline result: 70.67% test accuracy on 75 held-out images.** That test set is small enough that the number carries a confidence interval roughly twenty points wide, and [a dedicated section below](#read-this-before-quoting-the-accuracy) explains why that matters more than the point estimate does.

---

## Why This Matters

Children on the autism spectrum often experience the world of facial expressions differently from neurotypical children. Some have difficulty producing the facial expressions associated with specific emotions. Some have difficulty reading them in others. This creates a real, measurable gap in how existing emotion recognition systems (trained overwhelmingly on neurotypical faces like FER2013) perform when deployed with autistic users.

A model trained specifically on facial expressions of autistic children addresses two gaps at once:

1. **Representation.** Existing emotion recognition datasets rarely include autistic children. A model trained on FER2013 or AffectNet will predict poorly on the population it is supposed to serve in an assistive context
2. **Application.** There are genuine, ethical assistive-technology use cases for this kind of model — emotion-aware educational tools, therapist-in-the-loop communication aids, research into expression patterns across the spectrum

This project is an early-stage research baseline. It is not a deployed product. The [Ethical Considerations](#ethical-considerations) section later in this README is required reading before anyone considers extending this work toward real-world deployment.

---

## Published Work

The classifier in this repository is the visual emotion recognition component of the **Psycho-Intelligent Dialogue Agent (PIDA)** system, published at:

> Iwendi, C., Aboutorabi, N., Adesola, A. E., **Lemeke, C.**, Okoro, G. C., & Sharma, V. (2026). Psycho-Intelligent Dialogue Agents for Enhancing Emotional Self-Regulation in Autistic Teenagers. *2026 2nd IEEE International Conference on Cognitive Computing in Engineering, Communications, Sciences and Biomedical Health Informatics (IC3ECSBHI)*, pp. 658–663. DOI: 10.1109/IC3ECSBHI67834.2026.11468965

In that system, the classifier feeds a context-aware dialogue manager that adapts psychotherapeutic response strategies to the detected emotional state, with caregiver monitoring built in.

**A correction to the published paper.** The abstract of that paper describes the classifier as performing a *five*-emotion recognition task. That is an error we did not catch before publication. The model has **six** output classes — visible in `model.summary()` as `Dense(6)`, in the `emotion_labels` list in the notebook, and in the paper's own quoted chance level of 16.7%, which is one in six. This repository is the authoritative version.

---

## The Six Emotions

The model predicts one of six emotion classes, derived from the dataset's native labels:

| Index | Emotion |
|-------|---------|
| 0 | **Surprise** |
| 1 | **Delight** |
| 2 | **Sadness** |
| 3 | **Fear** |
| 4 | **Joy** |
| 5 | **Anger** |

Chance performance on a six-class balanced task is **16.7%**.

Worth noting: this six-class scheme diverges from the standard **Ekman-seven** (Happy, Sad, Angry, Fear, Disgust, Surprise, Neutral) that drives most facial expression datasets including FER2013. Three meaningful differences:

- **No "Disgust" class.** Disgust is the most consistently hardest-to-detect emotion in FER datasets even with large sample sizes. Its omission here is a reasonable curatorial choice
- **"Delight" and "Joy" as separate classes.** Most datasets collapse these into a single "Happy" class. Splitting them allows the model to distinguish between general positive affect (Joy) and intense momentary positive reactions (Delight). Whether this split is robust in practice is an empirical question — and one the error analysis in the published paper answers unfavourably, with Delight and Joy emerging as one of the two most-confused pairs
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

### Actual split sizes

These are the counts reported by `flow_from_directory` in the run committed to this repository:

| Split | Source | Images | Approx. per class |
|---|---|---|---|
| Train | 80% of `Train/` | **608** | ~101 |
| Validation | 20% of `Train/` | **150** | ~25 |
| Test | full `Test/` | **75** | ~12 |
| **Total** | | **833** | ~139 |

**This is a very small dataset**, and everything downstream should be read in that light. 608 training images across six classes is roughly one hundred examples per emotion — enough for transfer learning to work, not enough for the result to be stable. See [Read This Before Quoting the Accuracy](#read-this-before-quoting-the-accuracy).

Images are RGB photographs of autistic children, resized to **256×256** on load. The dataset preserves colour information (unlike grayscale datasets such as FER2013), which matters for transfer learning from ImageNet-pretrained weights that expect 3-channel input.

---

## Model Architecture

The network uses **VGG16 transfer learning** with a frozen backbone and a trainable custom classifier head.

```
Input (256, 256, 3)
│
├── VGG16 (frozen, ImageNet weights, include_top=False)
│   ├── Block 1: Conv+Conv+MaxPool           → (128, 128, 64)
│   ├── Block 2: Conv+Conv+MaxPool           → (64, 64, 128)
│   ├── Block 3: Conv+Conv+Conv+MaxPool      → (32, 32, 256)
│   ├── Block 4: Conv+Conv+Conv+MaxPool      → (16, 16, 512)
│   └── Block 5: Conv+Conv+Conv+MaxPool      → (8, 8, 512)
│
└── Custom Classifier Head (trainable) ─────────
    Flatten                       → (32,768,)
    Dense(512, ReLU)
    Dropout(0.5)
    Dense(256, ReLU)
    Dropout(0.3)
    Dense(6, Softmax)             → Output (6 classes)
```

### Parameter budget

| Layer | Output shape | Parameters |
|---|---|---|
| `vgg16` (Functional, frozen) | (None, 8, 8, 512) | 14,714,688 |
| `flatten` | (None, 32,768) | 0 |
| `dense` | (None, 512) | 16,777,728 |
| `dropout` | (None, 512) | 0 |
| `dense_1` | (None, 256) | 131,328 |
| `dropout_1` | (None, 256) | 0 |
| `dense_2` | (None, 6) | 1,542 |

| | Count | Size |
|---|---|---|
| **Total** | 65,446,484 | 249.66 MB |
| **Trainable** | 16,910,598 | 64.51 MB |
| **Non-trainable (frozen VGG16)** | 14,714,688 | 56.13 MB |
| Optimizer state | 33,821,198 | 129.02 MB |

Note the shape of this: **the first Dense layer alone holds 16.78M of the 16.91M trainable parameters — 99.2% of everything being learned.** That is a direct consequence of flattening an 8×8×512 feature map into a 32,768-dimensional vector before the first projection. It is also why the heaviest dropout (0.5) sits immediately after it. Replacing the Flatten with `GlobalAveragePooling2D` would cut trainable parameters to roughly 133K, a 127× reduction, and is the first item on the roadmap.

**Why transfer learning?**

Training from scratch on 608 images would almost certainly overfit, and the network would fail to learn useful low-level visual features (edges, textures, face parts) from the limited data available.

Instead, the ImageNet-pretrained convolutional backbone brings **pre-learned visual representations** — the same edge detectors, texture filters, and part detectors that let VGG16 recognise a thousand object categories. These features are transferable to facial expression classification almost for free. Only the classifier head is retrained from scratch to map those features to the six emotion classes.

**Why freeze the backbone?**

With a dataset this size, fine-tuning the full backbone risks catastrophic forgetting and severe overfitting. Freezing it means:

- Training is fast (roughly 16 seconds per epoch on a T4 after the first)
- The pretrained visual features are preserved
- The model can't overfit at the feature-extraction layer, only at the classifier

A natural next iteration (see [Roadmap](#roadmap)) would be a two-stage schedule: train the head first with the backbone frozen, then unfreeze the top VGG blocks and fine-tune at a very low learning rate.

---

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Input shape** | 256 × 256 × 3 (RGB) | Matches VGG16's expected input channels, preserves detail |
| **Batch size** | 64 (10 steps per epoch) | Standard for mid-size CNNs on T4 GPU |
| **Epochs** | 28 (no early stopping) | Fixed budget, tuned empirically |
| **Optimiser** | Adam (lr = 1×10⁻⁴) | Ten times below the default, because only the head is training |
| **Loss** | categorical_crossentropy | Standard for multi-class one-hot targets |
| **Validation split** | 0.2 (20% of `Train/`) | Holdout for monitoring |
| **Hardware** | Kaggle, 2× NVIDIA Tesla T4 | Free tier |

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

The brightness range and rotation together are the elements that matter for real-world robustness: they approximate the head tilt and lighting variation you get from a camera in a room rather than a studio.

---

## Pipeline Walkthrough

The notebook runs seven clearly numbered steps.

**Step 1 — Environment setup.** Imports TensorFlow, Keras layers, the VGG16 application, Adam, NumPy, Matplotlib and `ImageDataGenerator`. Defines the two dataset paths.

**Step 2 — Image size and augmentation.** Sets `img_size = (256, 256)` and `batch_size = 64`, then constructs the training generator with all augmentation parameters plus `validation_split=0.2`. This cell is the single biggest lever for generalisation on a dataset this small.

**Step 3 — Data loading.** Creates three `flow_from_directory` generators: train (`subset='training'`, augmented), validation (`subset='validation'`), and test (separate rescale-only generator over `Test/`). All use `class_mode='categorical'`.

**Step 4 — Model architecture.** Loads VGG16 with ImageNet weights and `include_top=False`, freezes every convolutional layer, and stacks the custom head. The output layer uses `len(train_generator.class_indices)` so the code is dataset-agnostic.

**Step 5 — Training.** Compiles with Adam at 1e-4 and runs 28 epochs. Produces accuracy and loss curves, and prints `model.summary()`.

**Step 6 — Evaluation.** Calls `model.evaluate(test_generator)` on the 75 held-out images and plots the result as a single bar.

**Step 7 — Predictions.** Two qualitative checks. A single-image prediction on `Test/fear/13.jpg` (correctly predicted as Fear in the committed run), and a 4×5 grid of 20 random test images with predicted labels and confidence scores.

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| **Backbone** | VGG16 with ImageNet weights | Transfers low-level visual features for free. Training from scratch on 608 images would severely overfit |
| **Freezing strategy** | Freeze the entire backbone, train only the head | Safest option for a small dataset. Fine-tuning needs careful schedules that are out of scope for a first baseline |
| **Input size** | 256 × 256 RGB | Matches VGG16's natural input scale and keeps full colour. Grayscale would waste the pretrained colour-sensitive filters |
| **Learning rate** | 1×10⁻⁴ | Ten times below the Adam default. Appropriate when only the head is trainable — large steps on a small parameter set cause oscillation |
| **Augmentation intensity** | Gentle (0.09 for shifts, shear, zoom) | Aggressive augmentation on a small specialised dataset risks distorting emotional features |
| **Dropout schedule** | 0.5 after Dense(512), 0.3 after Dense(256) | Heavier regularisation where 99% of the trainable parameters sit |
| **Number of classes** | 6 (native dataset labels) | Preserved from the curator's schema rather than mapped to Ekman-7. The curator's choice reflects domain expertise |
| **Output layer sizing** | Derived from `class_indices` | Works if the dataset later adds or removes classes |

---

## Results

All figures below are read from the stored outputs of the notebook committed to this repository.

### Test Set Performance

| Metric | Value |
|--------|-------|
| **Test accuracy** | **70.67%** (0.7067) |
| **Test loss** | 1.0723 |
| Test set size | 75 images |
| Approximate 95% confidence interval | **[60.4%, 81.0%]** |
| Chance level (6 classes) | 16.7% |

The model performs well above chance — roughly seven correct predictions in ten unseen images of autistic children, which is a reasonable result given the challenge of emotion recognition in this population and the size of the specialised dataset. The confidence interval is the honest companion to that number and is discussed [below](#read-this-before-quoting-the-accuracy).

### Training Trajectory

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|---|---|---|---|---|
| 1 | 0.3005 | 1.9503 | 0.4800 | 1.4046 |
| 5 | 0.4718 | 1.3691 | 0.5000 | 1.3029 |
| 10 | 0.5874 | 1.1636 | 0.5533 | 1.2552 |
| 15 | 0.6079 | 1.1100 | 0.5867 | 1.2923 |
| 20 | 0.5826 | 1.1152 | 0.5533 | 1.1668 |
| 24 | 0.6157 | 1.0519 | 0.5600 | **1.1396** (min) |
| 28 | **0.6644** | **0.9391** | 0.5800 | 1.1562 |

Training accuracy roughly doubled over the run, from 30.1% to 66.4%. Validation accuracy improved from 48.0% to 58.0%, with most of the gain arriving in the first fifteen epochs before flattening.

**Final state:** training 66.44%, validation 58.00%, test 70.67%.

The widening gap between training and validation in later epochs is mild overfitting — the model continued fitting the training data while generalisation plateaued. Validation loss reached its minimum at epoch 24 (1.1396) and rose slightly after, which is where early stopping would have halted the run. There is no early stopping in this version; the epoch budget is fixed at 28. Adding it is on the roadmap.

### Per-Class Performance

**The notebook does not currently compute a confusion matrix or per-class F1 scores.** This is the most significant gap in the evaluation and the top roadmap item.

What is known about per-class behaviour comes from the error analysis in the published paper: the residual errors concentrate in two pairs — **Surprise confused with Fear**, and **Delight confused with Joy**. Both pairs are ones humans find genuinely difficult, and the second is a direct challenge to the dataset's decision to split positive affect into two classes.

That finding is qualitative. Until per-class precision, recall and F1 are computed, the aggregate accuracy conceals whether any individual emotion is failing badly — exactly the problem quantified in the [companion FER2013 project](https://github.com/CollinsLemeke/Facial-Expression-Recognition-Model), where a headline figure hid a class the model recognised barely a third of the time.

### Context: Comparable Benchmarks

| Approach | Typical range |
|---|---|
| Chance (6 classes) | 16.7% |
| Simple CNN from scratch | 50–65% |
| **This work — frozen VGG16 transfer learning** | **70.7%** |
| Fine-tuned transfer learning, two-stage | 75–85% on better-curated datasets |

Direct comparison to FER2013 numbers is not meaningful: FER2013 has 35,887 images and uses the Ekman-seven scheme, while this dataset has 833 images and a different six-class schema.

---

## Read This Before Quoting the Accuracy

The test set is **75 images**. That is roughly twelve images per class, and it changes how the headline number should be read.

**One image is worth 1.33 accuracy points.** Getting two more images right moves the result from 70.67% to 73.33%. Getting two more wrong moves it to 68.00%. Nothing about the model changed in either case.

**The 95% confidence interval spans about twenty points.** Using the normal approximation, `0.7067 ± 1.96 × √(0.7067 × 0.2933 / 75)` gives roughly **[60.4%, 81.0%]**. Any comparison against another model whose result falls inside that band is not a meaningful comparison.

**This explains the test-above-validation result.** Test accuracy (70.67%) exceeded both validation (58.00%) and training (66.44%) accuracy, which looks anomalous. It is not evidence of unusually good generalisation. With 75 test images against 150 validation images, the test estimate is simply noisier, and this draw came out favourably. The published paper offers "test set representativeness" and "statistical variation" as explanations; the confidence interval is the precise version of that same point.

**What would fix it.** Stratified k-fold cross-validation over the combined 833 images, reporting a mean and standard deviation across folds rather than a single point estimate, plus per-class F1 so that minority-class failure becomes visible. That is the difference between "the model scored 70.67%" and "the model scores 70.67% ± *x*, and here is where it fails."

None of this makes the result worthless. It makes it *provisional*, which is the correct status for a baseline on a dataset this size. Quoting the point estimate without the interval would be the mistake.

---

## How to Reproduce

### Option 1: Run on Kaggle (Recommended)

1. Open [Kaggle](https://www.kaggle.com/) and sign in
2. Create a new notebook
3. Attach the dataset from the Kaggle data tab: search for `autistic-children-emotions-dr-fatma-m-talaat` and click **Add**
4. Upload `autism-emotion-detection.ipynb` or copy the code cells
5. Enable **GPU T4 x1** in notebook settings (free tier)
6. Run all cells top to bottom

Expected runtime on a T4: **roughly 10 minutes.** The first epoch takes about 73 seconds including XLA compilation and cuDNN warm-up; subsequent epochs run at around 16 seconds each.

**Note on reproducibility:** this notebook does not set a global random seed. Weight initialisation, shuffle order and augmentation draws will differ between runs, so your numbers will not match exactly. Given the confidence interval above, expect variation of several points. Adding `tf.keras.utils.set_random_seed(42)` before model construction is on the roadmap and would make runs comparable.

### Option 2: Run Locally

```bash
# Clone the repo
git clone https://github.com/CollinsLemeke/Autism-Facial-Emotion-Classification.git
cd Autism-Facial-Emotion-Classification

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Kaggle
# (You'll need a Kaggle API token — see https://www.kaggle.com/docs/api)
kaggle datasets download -d fatmam/autistic-children-emotions-dr-fatma-m-talaat
unzip autistic-children-emotions-dr-fatma-m-talaat.zip -d data/

# Update paths in the notebook from /kaggle/input/... to data/...
jupyter notebook autism-emotion-detection.ipynb
```

### Hardware Recommendations

- **Minimum:** CPU-only is impractical. Not recommended
- **Recommended:** Single modern GPU (T4, RTX 3060+, A10G). Around 10 minutes
- **Best:** A100 or L4. Under 5 minutes

---

## Ethical Considerations

This section is the most important part of this README. Emotion recognition on a clinical population requires careful framing.

### What This Project Is Not

- **Not a medical device.** This is research code. It is not validated, certified, or designed for clinical use
- **Not a diagnostic tool.** This model classifies facial expressions in images. It does not diagnose anything. Autism is not diagnosable from a photograph, and emotion is not inferable from a facial expression alone
- **Not a deployment-ready product.** Any deployment to real users, especially children with ASD, would require substantial additional work on fairness auditing, user consent frameworks, clinical oversight, and bias evaluation

### Dataset Limitations

- **Very small sample size.** 833 images total, 75 of them in the test set. Generalisation beyond the exact data distribution is genuinely uncertain, and the confidence interval on the headline result reflects that
- **Demographic representation.** The dataset was curated in a specific context. Performance on children outside that demographic (different regions, age groups, or positions on the spectrum) is unknown, and the dataset carries no demographic annotation with which to check
- **Label validity.** "Emotion" labels on facial expressions are always inferential. An image labelled "fear" captures a facial expression that a human annotator interpreted as fearful. The actual emotional state of the child at the moment of capture is unknowable
- **No neutral class.** The model always predicts one of six active emotions. A calm, unexpressive face will be forced into a category
- **Consent and privacy.** Images of children on the autism spectrum are sensitive. Any derivative work should honour the original consent framework under which the dataset was collected

### Considerations for Downstream Use

If anyone were to extend this work toward actual assistive technology, these would be non-negotiable:

- **Clinical collaboration.** Work directly with ASD clinicians, educators, and autistic self-advocates from day one, not retrofitted later
- **Autistic community involvement.** Nothing about us without us — the autistic community must be substantively involved in the design of any tool that uses this technology
- **Uncertainty communication.** The model is a probabilistic classifier. Its confidence scores must be surfaced to any downstream user, not hidden behind a confident-sounding label
- **Opt-in consent.** No passive deployment in classrooms, therapy sessions, or surveillance settings without active, informed consent from children (age-appropriately), parents, and care teams
- **Failure-mode design.** The system must fail safely — a wrong emotion prediction in an educational game has low stakes, the same prediction informing a clinical decision does not

### Broader Emotion Recognition Caveats

The same caveats that apply to general facial expression recognition apply here with added force due to the clinical population:

- **Emotion inference is not ground truth.** Models predict visual patterns an annotator labelled, not internal states
- **Cultural expression norms vary.** A model trained on one population's expression conventions may misread another's
- **Autistic expressions may differ from neurotypical norms.** This is partly why dedicated datasets like this one exist — but the training labels themselves are interpretations, possibly by neurotypical annotators, of autistic children's expressions. That interpretive layer should not be forgotten
- **Regulatory context.** Emotion inference systems are treated as high-risk or prohibited in several jurisdictions depending on setting. Article 5(1)(f) of Regulation (EU) 2024/1689 prohibits emotion inference in workplaces and education institutions except for medical or safety purposes. An assistive or therapeutic application may fall within the medical carve-out, but that is a determination requiring legal advice, not an assumption

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
│   ├── test_accuracy.png
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

In the order the results above actually justify:

1. **Confusion matrix and per-class precision, recall and F1.** The single most valuable addition. Aggregate accuracy on a six-class problem with ~12 test images per class cannot show which emotion is failing. The published error analysis suggests Surprise/Fear and Delight/Joy; this would confirm it quantitatively
2. **Cross-validation.** Stratified k-fold over the combined 833 images, reporting mean and standard deviation, replacing a single noisy point estimate
3. **Set a global seed.** `tf.keras.utils.set_random_seed(42)` before model construction, so runs are comparable
4. **Add early stopping.** Validation loss bottomed at epoch 24 and rose after. `EarlyStopping(patience=5, restore_best_weights=True)` would capture the better checkpoint automatically
5. **Replace Flatten with GlobalAveragePooling2D.** Cuts trainable parameters from 16.9M to roughly 133K, which on a 608-image training set is likely to help rather than hurt
6. **Two-stage fine-tuning** — train the head frozen, then unfreeze the top VGG blocks at 1e-5
7. **Modern backbones** — ResNet50, EfficientNet-B0, or a vision transformer at a similar parameter budget
8. **Face detection preprocessing** — MTCNN or MediaPipe to crop tightly before feature extraction
9. **Grad-CAM visualisation** — show which facial regions drive each prediction, essential for interpretability in a clinical context
10. **Cross-dataset evaluation** — test on FER2013 or AffectNet to quantify how much the specialised training actually matters
11. **Clinical validation study** — collaboration with ASD clinicians to evaluate whether predictions align with expert judgment

---

## Dataset Acknowledgement

This project uses the **Autistic Children Emotions dataset curated by Dr. Fatma M. Talaat**. Full credit for dataset collection, curation, and labelling belongs to Dr. Talaat. Anyone using the dataset should cite the curator's original work and comply with the dataset's licence terms as published on Kaggle.

---

## Author

**Collins Lemeke** — model design, implementation, training and evaluation.

AI Research Engineer, Centre of Intelligence of Things, University of Greater Manchester. This work was carried out with colleagues at CIoTh under the supervision of Prof. Celestine Iwendi, and forms part of a wider research programme on reading internal state from observable signals — across facial expression, physiological sensing, gait and language.

- [GitHub](https://github.com/CollinsLemeke)
- [Kaggle](https://www.kaggle.com/collinslemeke/code)

Related work in this programme:
- [Facial Expression Recognition with CNN](https://github.com/CollinsLemeke/Facial-Expression-Recognition-Model) — imbalance-aware evaluation on FER2013
- [Detecting Cognitive Decline, Falls and Frailty](https://github.com/CollinsLemeke/Detecting-Cognitive-Decline-Falls-and-Frailty) — interpretable screening from gait sensor data
- [DistilBERT vs Frontier LLMs](https://github.com/CollinsLemeke/DistilBERT-vs-Frontier-LLMs) — accuracy and carbon trade-offs on mental health text

For questions, feedback, or feature requests, open a GitHub issue.

---

## License

**Code: MIT.** Free to use, modify, and distribute. See [LICENSE](LICENSE) for full terms.

**Data:** The Autistic Children Emotions dataset has its own licence and terms of use, separate from this code, and is not redistributed here. Refer to the original Kaggle dataset page for licensing details.

---

> *Built with TensorFlow, Keras, and VGG16 transfer learning. Designed as a research baseline, not a deployed product. Read the Ethical Considerations section before extending this work.*
