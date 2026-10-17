# Model Findings (Member 1, AI Model)

Status: working notes for the report. Every number below was measured in this project; interpretations are marked as such.
Rule used throughout: the test sets were never used to choose or tune a model. Single training run per model (seed 42), so small differences are not reliable.

---

## 1. Data

Dataset: COVID-19 Radiography Database (Kaggle, tawsifurrahman), 4 classes, images 299x299, some stored as grayscale and some as RGB (converted to RGB at load time). All sampled images are visually grayscale (max channel difference 0 on 500 random test images).

| Class | Images (raw) | Images (after duplicate removal) |
|---|---|---|
| Normal | 10,192 | 10,191 |
| Lung_Opacity | 6,012 | 6,012 |
| COVID | 3,616 | 3,570 |
| Viral Pneumonia | 1,345 | 1,338 |
| Total | 21,165 | 21,111 |

- 54 exact duplicates (same MD5 hash) removed. Near-duplicates and same-patient images cannot be detected (no patient ID in the dataset).
- Imbalance: Normal is 7.6x larger than Viral Pneumonia. Accuracy is therefore not used alone; macro-F1 and per-class recall are the main metrics.
- Split: stratified 70/15/15, seed 42 -> Train 14,777 / Val 3,167 / Test 3,167 (identical class proportions in all three). Test: Normal 1,529, Lung_Opacity 902, COVID 535, Viral Pneumonia 201.
- Lung masks exist for all 21,111 images (256x256, binary, lungs cover on average 24% of the image, range 8% to 45%).

## 2. Training setup

- Transfer learning from ImageNet weights, final layer replaced by 4 outputs. Compared: ResNet50, DenseNet121, EfficientNet-B0.
- Input 224x224 RGB, ImageNet normalization. Augmentation (train only): rotation up to 10 degrees, translation up to 5%, scale 0.95 to 1.05, brightness/contrast 0.15. No horizontal flip (it would mirror the heart position).
- Class imbalance: cross-entropy with inverse-frequency class weights (COVID 1.478, Lung_Opacity 0.878, Normal 0.518, Viral Pneumonia 3.943).
- AdamW (lr 1e-4, weight decay 1e-4), batch 16, mixed precision, 8 epochs, cosine learning-rate schedule. Best epoch chosen by validation macro-F1.
- Hardware: laptop GPU with 4 GB VRAM. Time per epoch: EfficientNet-B0 about 104 s, ResNet50 about 144 s, DenseNet121 about 157 s.
- A first ResNet50 run without a learning-rate schedule (5 epochs) was unstable (validation macro-F1 between 0.876 and 0.936). The cosine schedule was added and the 8-epoch runs were stable.
- EfficientNet-B0 training curves: `analysis/training_curves_efficientnet_b0.png` (values copied by hand from the training log). Train loss fell from 0.406 to 0.077, validation loss from 0.172 to 0.120; train accuracy ended slightly above validation (0.964 vs 0.953), a mild gap.

## 3. Results on the original test set (3,167 images)

| Model | Accuracy | macro-F1 | macro ROC-AUC |
|---|---|---|---|
| ResNet50 | 0.959 | 0.969 | 0.9950 |
| EfficientNet-B0 | 0.955 | 0.964 | 0.9954 |
| DenseNet121 | 0.962 | 0.971 | 0.9950 |

Recall per class:

| Model | COVID | Lung_Opacity | Normal | Viral Pneumonia |
|---|---|---|---|---|
| ResNet50 | 0.985 | 0.928 | 0.965 | 0.985 |
| EfficientNet-B0 | 0.985 | 0.939 | 0.948 | 0.995 |
| DenseNet121 | 0.989 | 0.921 | 0.974 | 0.980 |

- The ranking on validation (EfficientNet-B0 best, macro-F1 0.959) differs from the ranking on test (DenseNet121 best). The differences are small, so the three models are practically equivalent here. The statistical comparison belongs to Member 3. The final model is NOT fixed yet; EfficientNet-B0 is used in the app as a provisional choice because it is the lightest and fastest.
- Lung_Opacity is the weakest class for all models. Viral Pneumonia has only 201 test images, so its metrics are less stable.
- These numbers are probably not representative of new hospitals (see sections 5 and 6).

## 3b. Variability across training seeds

Each model was trained 3 times with the same settings (seed 42 from the original run, plus seeds 1 and 2). Mean +- standard deviation over the 3 runs (the standard deviation of 3 runs is only a rough estimate):

| Model | Test accuracy | Test macro-F1 | External clean sensitivity | External clean specificity |
|---|---|---|---|---|
| EfficientNet-B0 | 0.958 +- 0.003 | 0.966 +- 0.002 | 0.989 +- 0.005 | 0.587 +- 0.054 |
| ResNet50 | 0.960 +- 0.001 | 0.970 +- 0.001 | 0.995 +- 0.003 | 0.559 +- 0.109 |
| DenseNet121 | 0.961 +- 0.001 | 0.970 +- 0.001 | 0.989 +- 0.008 | 0.551 +- 0.109 |

- Test accuracy differs by at most 0.3 points between models. EfficientNet-B0's macro-F1 was lower than ResNet50's and DenseNet121's in every run (0.964 to 0.968 vs 0.969 to 0.971), about 0.4 points on average; this is small and based on 3 runs per model.
- Specificity on the clean external set varies a lot between runs of the same model (EfficientNet-B0 0.533 to 0.640, ResNet50 0.434 to 0.632, DenseNet121 0.455 to 0.669), more than the differences between model means. The model ranking on external specificity in section 5 (one run each) is therefore not reliable. Sensitivity is stable (0.98 to 0.998).
- Test was evaluated once per run and never used to choose a checkpoint (the best epoch is chosen on validation).
- Seed 42 results come from the original training script, the other two from `model/experiments/train_seeds.py` with the same settings. GPU non-determinism was not controlled.

## 4. Explainability (Grad-CAM) and input robustness

- Grad-CAM (EfficientNet-B0, last feature layer, 7x7 map upsampled) is approximate, not pixel-accurate.
- Qualitative check on a handful of images: for COVID the heat is on both lung fields; for Normal it is on the heart and mid-chest; one Lung_Opacity example also showed heat on an image marker (top corner). A few images are not enough to conclude that the model uses markers.
- Color sensitivity (found by testing): the same COVID X-ray tinted blue was classified Normal with probability 1.000. Fix: `predict()` converts every input to grayscale before the model (as in the training data). After the fix the tinted image is classified COVID again. `predict()` also returns a `warnings` list that flags colored inputs. This was verified on two images only; robustness to other changes (camera photos, compression, rotation) was not tested.
- A random colored-noise image was classified Normal (0.572) and Lung_Opacity (0.428). The model has no "not an X-ray" option, and its confidence is not a reliable out-of-distribution signal.

## 5. External test: Chest X-Ray Images (Pneumonia), 2 classes

5,856 images (NORMAL 1,583; PNEUMONIA 4,273). Mapping: predicted Normal -> NORMAL; COVID, Lung_Opacity or Viral Pneumonia -> PNEUMONIA (an assumption: Lung_Opacity is not necessarily pneumonia, and the external pneumonia class includes bacterial cases the model was not trained on as a class).

Overlap check (downsampled-image correlation): 2,686 external images (46%) have a near-identical match in our main dataset (1,341 matched to Normal and 1,345 to Viral Pneumonia; all of them from the external "train" folder). The correlations separate cleanly (no image between 0.98 and 0.999), so the threshold is not borderline. Our Normal and Viral Pneumonia classes therefore partly come from this pediatric dataset.

| Model | Subset | Images | Sensitivity | Specificity (95% CI) | AUC |
|---|---|---|---|---|---|
| ResNet50 | all | 5,856 | 0.994 | 0.933 | 0.994 |
| ResNet50 | already in our data | 2,686 | 0.996 | 0.991 | 1.000 |
| ResNet50 | clean (not in our data) | 3,170 | 0.993 | 0.612 (0.549 to 0.671) | 0.962 |
| EfficientNet-B0 | clean | 3,170 | 0.995 | 0.533 (0.470 to 0.595) | 0.944 |
| DenseNet121 | clean | 3,170 | 0.981 | 0.669 (0.608 to 0.726) | 0.957 |

- The clean set has 242 healthy and 2,928 pneumonia images, so specificity has wide intervals.
- On clean images the models still flag pneumonia almost always but label 33% to 47% of healthy pediatric X-rays as abnormal (mostly "Viral Pneumonia").
- Clean healthy images all come from the external "test" and "val" folders (234 and 8). In that test folder specificity is 0.598 / 0.530 / 0.662 (ResNet50 / EfficientNet-B0 / DenseNet121).
- This is not an independent source: the clean images come from the same published collection as part of our training data. It measures unseen images from that collection, not a different hospital.
- Grad-CAM on 113 EfficientNet-B0 errors (healthy images predicted as Viral Pneumonia): the heat is on the lateral lung fields, not on markers or borders. Share of heat in the outer image band: 0.49 for errors, 0.26 for correctly classified healthy images, 0.50 for correctly classified pneumonia (crude measure, no lung mask).
- Not explained: why specificity falls so much on new pediatric images. Possible causes (untested): the model uses pediatric appearance as a shortcut for Viral Pneumonia; the external "test" folder differs from its "train" folder; real overlap between normal and pneumonia in children.

## 6. Patient overlap between our splits

File names in the pediatric collection contain an ID (assumed to identify the patient; not verified).

- Viral Pneumonia: 1,345 matched images from 1,090 IDs; 143 IDs have more than one image; 74 IDs are spread over more than one of our splits (236 images). 44 of the 201 Viral Pneumonia test images have an ID that also appears in our train split.
- Normal: 1,341 matched images from 1,211 IDs; 35 of the 212 matched test images share an ID with our train split.
- Effect on recall in our test split: no visible difference between images with and without an ID seen in training (Viral Pneumonia: 0.977 vs 0.987 ResNet50, 1.000 vs 0.994 EfficientNet-B0, 0.977 vs 0.981 DenseNet121; Normal: 1.000 vs 1.000, 0.971 vs 0.983, 1.000 vs 1.000). So there is no evidence that this overlap inflated test recall. Groups are small.
- COVID and Lung_Opacity have no patient ID, so overlap there could not be checked.
- Healthy pediatric images matched to our data are classified almost perfectly in our test split (212 images, recall 0.981 to 1.000) but poorly when they are new images from the same collection (section 5). This gap is unexplained.

## 7. Lung-mask experiments (diagnostic only)

Masks exist only for the main dataset, so these models cannot be used in the app (uploaded images have no mask).

| Experiment | Accuracy | macro-F1 |
|---|---|---|
| Original models on masked test images (no retraining): ResNet50 / EfficientNet-B0 / DenseNet121 | 0.356 / 0.296 / 0.271 | 0.329 / 0.279 / 0.270 |
| EfficientNet-B0 trained on masked images, tested on masked test | 0.932 | 0.935 |
| Same model on original (unmasked) test | 0.163 | 0.071 |
| Reference: EfficientNet-B0 on original test | 0.955 | 0.964 |

- The first row only shows that models fail on inputs unlike their training data; it says nothing about what they rely on.
- Training on masked images (lungs only) keeps most of the performance (about 2.3 points lower accuracy; COVID recall 0.985 -> 0.957, Lung_Opacity 0.939 -> 0.884). It was still improving at epoch 8, so part of the drop may be under-training.
- A classifier using only 12 numbers describing the mask shape (area, bounding box, centroid, spread; no image pixels) reaches accuracy 0.655 and macro-F1 0.647 (majority-class baseline 0.483), with Viral Pneumonia recall 0.876 and COVID recall 0.452. So lung shape alone carries class information, especially for Viral Pneumonia (consistent with pediatric images), and masking does not remove every non-disease clue. These 12 numbers are crude, so this is a lower bound on what shape contains.

## 8. Limitations to state in the report

1. The system assists radiologists and does not replace them. It is not clinically validated.
2. Public data from limited sources; no patient IDs; same-patient images may be in different splits (confirmed for part of two classes).
3. Normal and Viral Pneumonia classes partly come from a pediatric collection (children), the other classes appear to be adults (from visual inspection of a few images only). Age or source may act as a shortcut; not tested directly.
4. On unseen pediatric images specificity is low (about 0.53 to 0.67) while sensitivity stays high.
5. The model is sensitive to input changes (color was found and mitigated); no "not an X-ray" detection beyond a grayscale warning.
6. Grad-CAM is coarse and was inspected on few images.
7. One training run per model; differences of about one point are within noise.

## 9. Scripts (run from the repository root)

| Purpose | Script |
|---|---|
| EDA, duplicates, split | `model/prep/eda.py`, `model/prep/check_duplicates.py`, `model/prep/split_data.py` |
| Data pipeline, training, test evaluation, inference | `model/data.py`, `model/train.py`, `model/evaluate.py`, `model/predict.py` |
| Robustness checks | `model/tests/make_color_test.py`, `model/tests/check_gray.py` |
| External test and overlap | `model/experiments/evaluate_external.py`, `check_overlap.py`, `evaluate_external_clean.py`, `external_by_split.py`, `gradcam_errors.py`, `child_vs_adult_test.py` |
| Patient overlap | `model/experiments/patient_leak_check.py`, `patient_leak_effect.py` |
| Lung masks | `model/experiments/check_masks.py`, `mask_eval_notrain.py`, `train_masked.py`, `mask_silhouette_test.py` |
| Training curves | `analysis/plot_curves.py` |

Result files: `analysis/model_comparison_test.csv`, `analysis/external_generalization.csv`, `analysis/external_clean_generalization.csv`, `analysis/mask_eval_notrain.csv`, `analysis/mask_trained_eval.csv`. Prediction CSVs are in `data/predictions/` (not in GitHub; shared through Drive).