# Research Methodology

## 1. Research Objective

This project investigates whether combining URL-based features with HTML and webpage features improves phishing website detection compared with using URL-based features alone.

The research question is:

**Does combining URL-based features with HTML and webpage/body features improve cross-dataset phishing detection performance compared with using URL-based features alone?**

To investigate this question, two machine learning algorithms were evaluated using different feature configurations. The models were trained on one phishing dataset and tested on an independent dataset to assess their ability to generalize to previously unseen websites.

## 2. Datasets

Two datasets were used in this project.

### 2.1 CompPhish

CompPhish was used for training, cross-validation, and internal validation.

The dataset contains extracted URL and HTML features for legitimate and phishing websites.

After removing duplicate URLs, 15,351 records remained.

The dataset was divided into:

| Split | Records |
|---|---:|
| Training | 12,826 |
| Validation | 2,525 |
| Total | 15,351 |

A domain-aware split was performed using `GroupShuffleSplit` with `random_state=42`.

Websites were grouped by registered domain to prevent the same registered domain from appearing in both the training and validation sets.

The target labels were:

- `0`: Legitimate website
- `1`: Phishing website

### 2.2 PhreshPhish

PhreshPhish was used for independent cross-dataset evaluation.

A preliminary sample was used during pipeline development and validation. Those preliminary records were excluded from the final evaluation sample.

The final evaluation used 3,000 websites:

- 1,500 legitimate websites
- 1,500 phishing websites

The final sample was selected before evaluation, and its selection manifest was preserved to ensure reproducibility.

An additional domain-disjoint subset was constructed by excluding records whose registered domains appeared in the CompPhish training dataset.

| Evaluation subset | Records | Unique registered domains |
|---|---:|---:|
| Full external | 3,000 | 2,038 |
| Domain-disjoint | 2,246 | 1,811 |

The full external evaluation measures performance on an independent dataset, while the domain-disjoint evaluation provides a stricter assessment by removing registered-domain overlap with training data.

Domain-disjoint evaluation does not guarantee complete independence between phishing campaigns or website templates.

## 3. Feature Extraction

The models use features describing different characteristics of a website.

A total of 70 features were used:

| Feature category | Number of features | Examples |
|---|---:|---|
| URL-based | 47 | URL length, domain length, special characters, brand-related patterns |
| HTML-based | 23 | Forms, links, images, scripts, external resources |
| Total | 70 | |

The original CompPhish feature extraction implementation was adapted into a reusable Python pipeline.

The pipeline accepts a URL and its corresponding HTML content, extracts the required features, and returns values in a consistent format.

### 3.1 Feature Validation

Feature validation was performed to check consistency with the original feature extraction implementation.

The validation process included:

- Comparing reconstructed URL features against existing CompPhish feature values.
- Testing brand-related feature extraction.
- Testing HTML features using controlled HTML examples.
- Comparing the reconstructed HTML calculations with the original extractor.
- Checking that the combined pipeline returned all 70 expected features.

Most URL features matched the original dataset values. A small number of discrepancies were identified in threshold-dependent and brand-related features.

HTML extraction was verified through controlled consistency tests, although the original historical HTML snapshots were not available for complete record-by-record verification.

These limitations were documented rather than modifying the feature definitions based on external evaluation performance.

### 3.2 Training-Derived Thresholds

Some URL features depend on dataset-level thresholds, including:

- `url_more_than_mean`
- `domain_more_than_mean`

These thresholds were calculated using CompPhish training data and preserved in `training_thresholds.json`.

For the final external evaluation, the same training-derived thresholds were applied to PhreshPhish.

This prevented external evaluation records from influencing the feature thresholds.

## 4. Machine Learning Models

Two classification algorithms were evaluated.

### 4.1 Logistic Regression

Logistic Regression was used as a linear classification baseline.

Configuration:

- `StandardScaler` preprocessing
- `max_iter=2000`
- `random_state=42`

Scaling was included in a scikit-learn pipeline so that the scaler could be fitted using training data only.

### 4.2 Random Forest

Random Forest was used as a nonlinear ensemble classification model.

Configuration:

- `n_estimators=300`
- `random_state=42`
- `n_jobs=-1`

### 4.3 Feature Configurations

Three configurations were investigated during development:

1. **URL-only:** Uses 47 URL features.
2. **HTML-only:** Uses 23 HTML features.
3. **URL + HTML:** Uses all 70 features.

The main research comparison focused on URL-only and URL + HTML.

Both algorithms were evaluated using the same feature configurations.

The classification threshold was 0.5.

## 5. Experimental Design

The evaluation was organized into three main stages:

1. Internal validation using CompPhish.
2. Five-fold grouped cross-validation using CompPhish training data.
3. Independent cross-dataset evaluation using PhreshPhish.

### 5.1 Internal Validation

The CompPhish training set was used to fit the models.

The separate CompPhish validation set was used to evaluate model performance on unseen registered domains within the same dataset.

This provided a baseline for comparing internal performance with external performance.

### 5.2 Five-Fold Grouped Cross-Validation

Five-fold cross-validation was performed using `StratifiedGroupKFold` with:

- `n_splits=5`
- `shuffle=True`
- `random_state=42`

The cross-validation experiment used the CompPhish training set, leaving the separate validation set unchanged.

Websites were grouped by registered domain. Each validation fold contained domains that did not appear in that fold's training portion.

For every fold:

1. The training and validation portions were identified.
2. Fold-specific URL length and domain length thresholds were calculated using the fold's training records.
3. The threshold-dependent features were updated for that fold.
4. A new model was trained.
5. Predictions were generated for the validation portion.
6. Evaluation metrics were recorded.

For Logistic Regression, feature scaling was fitted separately within each training fold.

The experiment used four model configurations across five folds, resulting in 20 model fits.

The mean and standard deviation of each evaluation metric were calculated across the five folds.

**Cross-validation results**

| Model | Features | Mean F1 | F1 Standard Deviation | Mean ROC-AUC |
|---|---|---:|---:|---:|
| Logistic Regression | URL-only | 0.9007 | 0.0375 | 0.9668 |
| Logistic Regression | URL + HTML | 0.9347 | 0.0324 | 0.9855 |
| Random Forest | URL-only | 0.9338 | 0.0110 | 0.9801 |
| Random Forest | URL + HTML | 0.9595 | 0.0100 | 0.9932 |

The combined feature configuration achieved higher mean F1-scores for both algorithms.

The improvement was observed in all five folds.

The URL and domain threshold features were recalculated within each fold. Their exact implementation should remain consistent with the original feature extractor.

### 5.3 Independent Cross-Dataset Evaluation

The final models were trained using CompPhish and frozen before evaluation on the final PhreshPhish sample.

The external dataset was not used to retrain the frozen models or optimize classification thresholds.

The same feature extraction pipeline and training-derived thresholds were used.

Two external evaluations were conducted:

**Full external evaluation:** All 3,000 selected PhreshPhish samples.

**Domain-disjoint evaluation:** The 2,246 records remaining after registered-domain overlap filtering.

This design allowed the project to investigate both cross-dataset performance and the effect of removing overlapping registered domains.

## 6. Evaluation Metrics

Six metrics were used to evaluate model performance.

| Metric | Purpose |
|---|---|
| Accuracy | Percentage of correctly classified websites |
| Precision | Proportion of predicted phishing websites that were phishing |
| Recall | Proportion of actual phishing websites correctly detected |
| F1-score | Harmonic mean of precision and recall |
| ROC-AUC | Ability to distinguish between classes across thresholds |
| PR-AUC | Precision-recall performance across thresholds |

F1-score was the primary comparison metric because both false positives and false negatives are important in phishing detection.

ROC-AUC and PR-AUC were also reported to assess model discrimination beyond a single classification threshold.

## 7. Final Cross-Dataset Results

### 7.1 Full External Evaluation

The final evaluation included 3,000 PhreshPhish records.

| Model | Features | Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|
| Logistic Regression | URL-only | 0.6550 | 0.6935 | 0.7316 | 0.7548 |
| Logistic Regression | URL + HTML | 0.7997 | 0.8052 | 0.8701 | 0.8602 |
| Random Forest | URL-only | 0.6433 | 0.6984 | 0.7040 | 0.7141 |
| Random Forest | URL + HTML | 0.8190 | 0.8186 | 0.8767 | 0.8534 |

Both algorithms achieved higher F1-scores using combined URL and HTML features.

Random Forest with combined features achieved the highest F1-score of 0.8186.

### 7.2 Domain-Disjoint Evaluation

The domain-disjoint evaluation contained 2,246 samples.

| Model | Features | Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|
| Logistic Regression | URL-only | 0.6331 | 0.6712 | 0.6791 | 0.6998 |
| Logistic Regression | URL + HTML | 0.7640 | 0.7739 | 0.8231 | 0.8100 |
| Random Forest | URL-only | 0.6180 | 0.6779 | 0.6273 | 0.6245 |
| Random Forest | URL + HTML | 0.7792 | 0.7817 | 0.8310 | 0.7944 |

The combined models continued to outperform URL-only models after excluding overlapping registered domains.

The strongest domain-disjoint F1-score was 0.7817, achieved by Random Forest with combined features.

## 8. Statistical Analysis

Domain-grouped bootstrap resampling was used to estimate uncertainty in the F1-score improvements.

For each comparison, the difference between combined-feature and URL-only F1-scores was calculated using paired predictions on the same evaluation records.

Bootstrap sampling was performed at the registered-domain level to account for websites sharing the same domain.

A total of 5,000 bootstrap resamples were used.

### Results

| Evaluation | Model | F1 Improvement | 95% Confidence Interval |
|---|---|---:|---|
| Full external | Logistic Regression | +0.1117 | [0.0810, 0.1491] |
| Full external | Random Forest | +0.1202 | [0.0934, 0.1496] |
| Domain-disjoint | Logistic Regression | +0.1027 | [0.0853, 0.1204] |
| Domain-disjoint | Random Forest | +0.1038 | [0.0834, 0.1254] |

All four confidence intervals were above zero.

This supports the conclusion that combining URL and HTML features produces a consistent improvement in F1-score under the tested conditions.

## 9. Dataset Shift

An important finding was the decrease in performance between internal CompPhish evaluation and external PhreshPhish evaluation.

Although the combined models achieved high F1-scores during internal validation and cross-validation, their performance decreased on the independent dataset.

Feature distribution analysis also identified differences between CompPhish and PhreshPhish, including URL length characteristics, token counts, and other URL-based properties.

These findings suggest that dataset shift affects phishing model generalization.

Adding HTML features improves performance but does not eliminate the challenges of transferring models between datasets.

## 10. Backend Implementation

A Python backend was developed using FastAPI to demonstrate the trained phishing detection models.

Two frozen Random Forest models are used:

- **URL-only model:** Uses 47 URL features.
- **URL + HTML model:** Uses all 70 features.

The backend provides a `/predict` endpoint that accepts website information in JSON format.

For each request, the backend:

1. Validates the input.
2. Extracts the required features.
3. Selects the appropriate trained model.
4. Generates a classification.
5. Returns the predicted label and model-estimated phishing score.

The frontend is maintained in a separate React repository and communicates with the backend through HTTP requests.

The backend currently accepts supplied URL and HTML content. Automatic retrieval of webpage HTML is not part of the current inference endpoint.

## 11. Reproducibility

The project preserves the scripts and artifacts used to produce the experimental results.

Important files include:

| File or directory | Purpose |
|---|---|
| `research/preprocessing/` | Dataset preparation |
| `research/evaluation/` | Model evaluation and cross-validation |
| `research/validation/` | Feature validation |
| `research/final_experiment/` | Final external experiment |
| `phishing_pipeline/` | Feature extraction implementation |
| `data/training_thresholds.json` | Training-derived feature thresholds |
| `data/final_experiment/final_manifest.csv` | Final sample selection |
| `results/grouped_cv_folds.csv` | Individual cross-validation fold metrics |
| `results/grouped_cv_summary.csv` | Cross-validation summary |
| `results/final_experiment/final_metrics.csv` | Final evaluation metrics |
| `results/final_experiment/final_grouped_confidence_intervals.csv` | Bootstrap results |

The final models were saved before external evaluation, and the final sample selection was preserved.

## 12. Limitations

Several limitations should be considered when interpreting the results.

- Some reconstructed URL feature values differed from the original CompPhish dataset.
- Historical HTML snapshots were unavailable for complete record-level verification of HTML features.
- The cross-validation experiment used previously extracted features, with fold-specific corrections applied to two threshold-dependent features.
- Registered-domain separation does not guarantee independence between phishing campaigns or related websites.
- The final external evaluation used a selected subset of PhreshPhish rather than the complete dataset.
- Model performance decreased substantially between the training dataset and the external dataset.
- The backend's phishing score is not a calibrated probability and should not be interpreted as a guarantee that a website is safe.

## 13. Conclusion

The results support the hypothesis that combining URL and HTML features improves cross-dataset phishing website detection compared with URL-based features alone.

The improvement was observed for both Logistic Regression and Random Forest during grouped cross-validation, internal evaluation, and independent cross-dataset evaluation.

Domain-grouped bootstrap confidence intervals further supported the improvements observed on PhreshPhish.

However, the performance decrease between CompPhish and PhreshPhish highlights the importance of dataset shift and the remaining challenges of building phishing detection systems that generalize reliably to unfamiliar websites.