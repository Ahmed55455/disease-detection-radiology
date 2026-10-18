# Report draft: Member 1 (AI Model), sections C and D

Draft text for the team report. Numbers come from `docs/model_findings.md`. Items in [brackets] are placeholders for a figure, a table or a citation to add.

---

## C. Explainability and Robustness

### C.1 Grad-CAM

To show which image regions influenced a prediction, we used Grad-CAM [citation: Selvaraju et al., 2017] on the last convolutional feature layer of EfficientNet-B0. The feature map has a resolution of 7x7 and is upsampled to 224x224, so the heat maps are coarse and show approximate regions, not exact boundaries. They are an aid for the radiologist and not a proof of the model's reasoning.

On a small number of examples, the heat for COVID cases lay over both lung fields, and for Normal cases over the heart and mid-chest region. One Lung Opacity example also showed heat near an image marker in a corner of the image. A few images are not enough to conclude that the model uses such markers [Figure C1: example heat maps, `demo_images/` or `analysis/`].

### C.2 Sensitivity to input color

A robustness test showed that the model can be fooled by a change that has no medical meaning. The same COVID X-ray, tinted blue, was classified as Normal with probability 1.000. The cause is that the model was trained only on grayscale images. We fixed this by converting every input to grayscale before the model, and after the fix the tinted image is classified as COVID again. The prediction function also returns a warning when the uploaded image contains color, and the application shows it to the user.

This was verified on two images. Other input changes (camera photos of films, compression, rotation, different image sizes) were not tested and may also change the predictions.

### C.3 Inputs that are not X-rays

The model always outputs one of its four classes, even for an image that is not a chest X-ray. A randomly generated noise image was assigned to a disease class with a confidence of 0.465 to 0.572 in two tests (the class differed between the two random images). The model's confidence is therefore not a reliable signal that the input is valid, and the model also gave a confident wrong answer on a real test image (a Normal X-ray classified as Lung Opacity with confidence 1.000). In the application, this is why the final decision stays with the radiologist, and the color warning is only a partial safeguard.

---

## D. External Validation and Limitations

### D.1 External dataset

To test how the models behave on images they never saw, we used the public Chest X-Ray Images (Pneumonia) dataset [citation: Kermany et al., 2018; Kaggle: paultimothymooney/chest-xray-pneumonia], which has two classes (NORMAL and PNEUMONIA) and 5,856 images. To compare it with our four-class models, a predicted Normal was mapped to NORMAL and a predicted COVID, Lung Opacity or Viral Pneumonia was mapped to PNEUMONIA. This mapping is an assumption: Lung Opacity is not necessarily pneumonia, and the external pneumonia class includes bacterial cases that our models were not trained on as a class.

### D.2 Overlap with the training data

Before interpreting the external results, we checked whether the two datasets share images, by comparing downsampled versions of the images. 2,686 of the 5,856 external images (46%) had a near-identical match in our main dataset (1,341 matched to our Normal class and 1,345 to our Viral Pneumonia class), and all of them were in the external "train" folder. The matches separated clearly (no image had a similarity between 0.98 and 0.999). Our Normal and Viral Pneumonia classes therefore partly come from this pediatric collection.

This changes the interpretation of the external results (Table D1). Over all images, specificity looks high (0.916 to 0.947), but this is inflated by the images that were already in our training data. On the 3,170 images that are not in our data (242 healthy and 2,928 pneumonia), sensitivity stays high but specificity falls to between 0.53 and 0.67.

**Table D1. External results, original runs (seed 42)**

| Model | Subset | Sensitivity | Specificity (95% CI) | AUC |
|---|---|---|---|---|
| ResNet50 | all images | 0.994 | 0.933 | 0.994 |
| ResNet50 | not in our data | 0.993 | 0.612 (0.549 to 0.671) | 0.962 |
| EfficientNet-B0 | all images | 0.995 | 0.916 | 0.990 |
| EfficientNet-B0 | not in our data | 0.995 | 0.533 (0.470 to 0.595) | 0.944 |
| DenseNet121 | all images | 0.983 | 0.947 | 0.993 |
| DenseNet121 | not in our data | 0.981 | 0.669 (0.608 to 0.726) | 0.957 |

Across the three training runs per model, specificity on the images not in our data ranged from 0.43 to 0.67 for the same architecture, so the ranking of the models on this metric is not reliable. Sensitivity was stable (0.98 to 1.0). In practice, the models almost never miss pneumonia on these images, but they label roughly one third to one half of healthy children's X-rays as abnormal.

This external set is not fully independent: its images come from the same published pediatric collection that supplied part of our training data. It measures unseen images of that collection, not images from a different hospital. We did not tune any model or threshold on it.

We did not find the cause of the low specificity. Possible explanations, none of them tested, are that the model uses the typical appearance of a child's X-ray as a shortcut for Viral Pneumonia, that the external "test" folder differs from its "train" folder, or that there is a genuine overlap between normal and pneumonia images in children. A Grad-CAM analysis of 113 wrongly classified healthy images showed heat on the lateral lung fields, not on image markers or borders.

### D.3 Patient overlap between our splits

File names in the pediatric collection contain an identifier, which we assume refers to the patient. For the Viral Pneumonia images we could identify, 1,345 images came from 1,090 identifiers, and 74 identifiers appeared in more than one of our splits. 44 of the 201 Viral Pneumonia test images (and 35 of 212 matched Normal test images) had an identifier that also appears in our training set. Recall on these images was not different from recall on images with a new identifier (for example, for Viral Pneumonia 0.977 versus 0.987 for ResNet50), so we found no evidence that this overlap inflated the test results. The groups are small, and for COVID and Lung Opacity there is no patient identifier, so the overlap could not be checked.

### D.4 Lung-mask experiments

The dataset provides lung masks. Because uploaded images in the application have no mask, we used them only as diagnostic experiments. A model trained on lung-only images reached an accuracy of 0.932 (macro-F1 0.935) on masked test images, compared with 0.955 for the unmasked model, so most of the performance remains when everything outside the lungs is removed. However, a simple classifier that used only 12 numbers describing the shape of the mask (area, bounding box, position) and no image pixels reached an accuracy of 0.655 (macro-F1 0.647; the majority-class baseline is 0.483), with a Viral Pneumonia recall of 0.876. This means the lung outline alone carries information about the class, especially for Viral Pneumonia, so masking does not remove every non-disease clue.

### D.5 Limitations

1. **Assistive use only.** The system assists radiologists and does not replace them. It has not been clinically validated.
2. **Limited data sources.** The training data are public datasets from limited sources. The main dataset has no patient identifier, so images of the same patient may be in different splits (confirmed for part of two classes, with no measurable effect on recall).
3. **Possible age or source shortcut.** The Normal and Viral Pneumonia classes partly come from a pediatric collection, and the other classes appear to be adults (judged from a few images only). The model may use age or source as a shortcut. This was not tested directly.
4. **Low and unstable specificity on new pediatric images**, with high sensitivity (section D.2).
5. **Input sensitivity.** The model reacted to a color change (mitigated), and there is no detection of images that are not chest X-rays beyond a color warning.
6. **Coarse explanations.** Grad-CAM is low-resolution and was inspected on few images.
7. **Close model performance.** Differences of about one percentage point between the three architectures are within or near the variation between training runs.
8. **Confidence is not correctness.** The model can be wrong with a confidence of 1.000.