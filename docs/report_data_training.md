# Report draft: Member 1 (AI Model), sections A and B

Draft text for the team report. Numbers come from `docs/model_findings.md`. Items in [brackets] are placeholders for a figure, a table or a citation to add.

---

## A. Data and Preprocessing

### A.1 Dataset

We used the COVID-19 Radiography Database [citation: Chowdhury et al., 2020; Rahman et al., 2021; Kaggle: tawsifurrahman/covid19-radiography-database], a public collection of chest X-ray images in four classes: COVID, Lung Opacity, Normal and Viral Pneumonia. The images are 299x299 pixels. Some files are stored as grayscale and some as three-channel files, but the sampled images contain no color information (the difference between color channels was 0 on 500 random test images). All images are converted to three channels when loaded.

The dataset has 21,165 images, with strong class imbalance (Table A1). Normal is 7.6 times larger than Viral Pneumonia. Because of this imbalance, accuracy alone is a misleading metric, so we report macro-averaged F1 and per-class recall as the main metrics.

**Table A1. Class counts before and after duplicate removal**

| Class | Raw | After duplicate removal |
|---|---|---|
| Normal | 10,192 | 10,191 |
| Lung Opacity | 6,012 | 6,012 |
| COVID | 3,616 | 3,570 |
| Viral Pneumonia | 1,345 | 1,338 |
| Total | 21,165 | 21,111 |

### A.2 Quality control

Exact duplicate files were found by MD5 hash and removed (54 images). Near-duplicates and images of the same patient cannot be detected in the main dataset because it contains no patient identifier. This is a limitation (see section [Limitations]): images of the same patient may appear in both the training and the test split, which could make the test results more optimistic than performance on new patients.

### A.3 Data split

The data were split into training, validation and test sets with stratified sampling (70% / 15% / 15%, random seed 42), so that each split keeps the same class proportions (Table A2). The test set was used only for final reporting. The best epoch of each training run was selected on the validation set, and no model or setting was chosen using the test set.

**Table A2. Split sizes**

| Split | Images |
|---|---|
| Train | 14,777 |
| Validation | 3,167 |
| Test | 3,167 (Normal 1,529; Lung Opacity 902; COVID 535; Viral Pneumonia 201) |

The Viral Pneumonia class has only 201 test images, so its per-class metrics have wider uncertainty than those of the other classes.

### A.4 Preprocessing and augmentation

All images are resized to 224x224 pixels, converted to three channels and normalized with the ImageNet mean and standard deviation, because the networks were pre-trained on ImageNet. During training only, we applied random rotation (up to 10 degrees), translation (up to 5%), scaling (0.95 to 1.05) and brightness and contrast changes (0.15). Horizontal flipping was deliberately not used, because it would mirror the position of the heart and create anatomically unrealistic images.

At inference time, every input is converted to grayscale before the model. This was added after a robustness test showed that tinting a COVID X-ray blue changed the prediction to Normal with probability 1.000. After the change the tinted image is classified correctly again. The prediction function also returns a warning when the input image contains color. This was verified on two images only, and robustness to other changes (camera photos, compression, rotation) was not tested.

---

## B. Model Training and Comparison

### B.1 Architectures

We compared three convolutional networks pre-trained on ImageNet: ResNet50 [citation: He et al., 2016], DenseNet121 [citation: Huang et al., 2017] and EfficientNet-B0 [citation: Tan and Le, 2019]. The final classification layer of each network was replaced by a layer with four outputs. All layers were fine-tuned (transfer learning).

### B.2 Training settings

All models used the same settings: AdamW optimizer (learning rate 1e-4, weight decay 1e-4), cosine learning-rate schedule, batch size 16, mixed-precision training and 8 epochs. To handle class imbalance we used a cross-entropy loss with inverse-frequency class weights (COVID 1.478, Lung Opacity 0.878, Normal 0.518, Viral Pneumonia 3.943). The checkpoint with the highest validation macro-F1 was kept. Training ran on a laptop GPU with 4 GB of memory; one epoch took about 105 to 135 seconds for EfficientNet-B0, 145 seconds for ResNet50 and 160 seconds for DenseNet121.

A first ResNet50 run without a learning-rate schedule was unstable (validation macro-F1 between 0.876 and 0.936 over 5 epochs), so the cosine schedule was introduced. The EfficientNet-B0 training curves show a steady decrease in both training loss (0.406 to 0.077) and validation loss (0.172 to 0.120), with a small gap between training and validation accuracy (0.964 and 0.953) [Figure B1: `analysis/training_curves_efficientnet_b0.png`].

### B.3 Results on the test set

**Table B1. Test results of the original runs (seed 42)**

| Model | Accuracy | Macro-F1 | Macro ROC-AUC |
|---|---|---|---|
| ResNet50 | 0.959 | 0.969 | 0.9950 |
| EfficientNet-B0 | 0.955 | 0.964 | 0.9954 |
| DenseNet121 | 0.962 | 0.971 | 0.9950 |

**Table B2. Recall per class on the test set (seed 42)**

| Model | COVID | Lung Opacity | Normal | Viral Pneumonia |
|---|---|---|---|---|
| ResNet50 | 0.985 | 0.928 | 0.965 | 0.985 |
| EfficientNet-B0 | 0.985 | 0.939 | 0.948 | 0.995 |
| DenseNet121 | 0.989 | 0.921 | 0.974 | 0.980 |

[Figure B2: confusion matrices, `analysis/confusion_*.png`]

All three models reach about 96% accuracy and a macro ROC-AUC above 0.99. Lung Opacity is the hardest class for every model.

### B.4 Variability between training runs

The ranking on the validation set (EfficientNet-B0 first) differed from the ranking on the test set (DenseNet121 first), which suggested that the differences are small. To measure this, each model was trained three times (seeds 42, 1 and 2).

**Table B3. Mean and standard deviation over three runs per model**

| Model | Test accuracy | Test macro-F1 |
|---|---|---|
| EfficientNet-B0 | 0.958 +- 0.003 | 0.966 +- 0.002 |
| ResNet50 | 0.960 +- 0.001 | 0.970 +- 0.001 |
| DenseNet121 | 0.961 +- 0.001 | 0.970 +- 0.001 |

Test accuracy differs by at most 0.3 percentage points between models. EfficientNet-B0 had a lower macro-F1 than the other two models in every run (about 0.4 points on average), but with three runs per model this is not conclusive. We therefore consider the three architectures practically equivalent on this test set, and the choice of the model used in the application depends on practical factors (size and speed) and on the statistical comparison performed by the simulation and statistics member of the team. EfficientNet-B0, the lightest model, is used as a provisional choice in the application.

Note: GPU non-determinism was not controlled, and three runs give only a rough estimate of the spread.

### B.5 What these results do and do not show

The test set comes from the same dataset as the training data, so these results describe performance on similar images, not performance in a new hospital. On a separate pediatric pneumonia dataset (two classes), sensitivity stayed high (about 0.98 to 1.0) but specificity on images not present in our data was low and varied between training runs (about 0.43 to 0.67) [see section External Validation]. The system is intended to assist radiologists and does not replace them.