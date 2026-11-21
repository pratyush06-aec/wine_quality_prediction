# Wine Quality Prediction

## Overview

This project is a small, reproducible machine-learning experiment that predicts whether a wine is likely to be **good quality** or **bad quality** from its physicochemical measurements.

The source dataset combines red and white wine observations. The application:

1. Loads `wine_quality_merged.csv`.
2. Removes the categorical `type` column.
3. Converts the original quality score into a binary target.
4. Splits the data into training and test sets.
5. Trains a `RandomForestClassifier`.
6. Prints accuracy, precision, recall, F1 score, the confusion matrix, and a full classification report.

The implementation is intentionally compact and currently contained in [main.py](main.py).

---

## Problem statement

Wine quality in the source data is represented by an integer score rather than only the categories “good” and “bad”. This project reformulates the task as binary classification:

| Mapped class | Original quality score |
|---|---|
| `0` — bad/acceptable quality | 3, 4, or 5 |
| `1` — good quality | 6, 7, 8, or 9 |

The threshold is `quality >= 6`. The value `6` therefore belongs to the positive class. This is an explicit modeling decision and means that “good” does not necessarily mean the very highest sensory quality.

## Dataset

The project uses [wine_quality_merged.csv](wine_quality_merged.csv).

- Total observations: **6,497**
- Input columns: **11 physicochemical features**
- Target: `quality`
- Wine type: `type`, with values `red` or `white`
- Missing values: **0**
- Duplicate rows: **1,177** under an exact-row comparison
- Red wine observations: **1,599**
- White wine observations: **4,898**

### Features

| Feature | Meaning |
|---|---|
| `fixed acidity` | Fixed (non-volatile) acids in the wine |
| `volatile acidity` | Acetic-acid-related acidity |
| `citric acid` | Citric acid content |
| `residual sugar` | Sugar remaining after fermentation |
| `chlorides` | Salt concentration |
| `free sulfur dioxide` | Unbound sulfur dioxide |
| `total sulfur dioxide` | Free plus bound sulfur dioxide |
| `density` | Wine density |
| `pH` | Acidity/basicity measure |
| `sulphates` | Sulphate concentration |
| `alcohol` | Alcohol percentage |

`type` is removed before training because the current model would otherwise require adding an encoding strategy for red/white labels. The project therefore evaluates a wine-quality model based on numeric chemical measurements only.

### Quality distribution

| Quality | Count |
|---|---:|
| 3 | 30 |
| 4 | 216 |
| 5 | 2,138 |
| 6 | 2,836 |
| 7 | 1,079 |
| 8 | 193 |
| 9 | 5 |

The original multiclass target is strongly concentrated in scores `5` and `6`. After binary mapping:

- Bad/acceptable (`0`): **2,384** observations
- Good (`1`): **4,113** observations

This means the binary target is **moderately imbalanced**, with the positive class representing about 63% of the observations.

### Feature analysis

The source code contains commented placeholders for distribution and correlation plots. Pearson correlation against the original integer quality score shows:

- Strongest positive relationship: `alcohol` (**0.4443**)
- Strongest negative relationships: `density` (**-0.3059**) and `volatile acidity` (**-0.2657**)
- Chlorides also correlate negatively (**-0.2007**)

Correlation describes linear association, not causation. It was not used to manually select features: the random forest receives all 11 numeric features.

---

## Project structure

```text
wine_quality_prediction/
|-- main.py                 # Data loading, preprocessing, model training and evaluation
|-- wine_quality_merged.csv # Combined red and white wine dataset
|-- requirements.txt        # Python dependencies
|-- README.md               # Project documentation
`-- .venv/                  # Optional local virtual environment
```

---

## Modeling decisions

### 1. Binary label conversion

The original quality scores do not need to be predicted as exact values for this application. The code applies:

```python
y = dataframe["quality"].apply(lambda value: 1 if value >= 6 else 0)
```

This creates a single, interpretable classification target and avoids a highly sparse nine-class problem.

**Trade-off:** information is lost. Scores `6`, `7`, `8`, and `9` are all considered “good,” and scores `3`, `4`, and `5` are all “bad/acceptable.” A future exact-quality model could preserve more granularity.

### 2. Wine type removal

The categorical `type` column is dropped:

```python
dataframe = dataframe.drop(columns=["type"])
```

**Reason:** all remaining model inputs are numeric and can be used directly by scikit-learn. Encoding red/white as numeric values was avoided because it could add an ordinal interpretation that has no meaningful numeric order.

**Trade-off:** a wine model can benefit from learning separate patterns for red and white wine, so retaining and one-hot encoding `type` is a possible improvement.

### 3. Train/test split

The project uses an 80/20 split:

```python
train_test_split(x, y, test_size=0.2, random_state=2)
```

- Training observations: **5,197**
- Test observations: **1,300**
- `random_state=2` makes the row assignment reproducible

A fixed random seed is used for repeatable demonstrations, though model retraining should still be validated against an unseen dataset or cross-validation.

### 4. Random forest classifier

The model is scikit-learn's default `RandomForestClassifier`:

```python
model = RandomForestClassifier()
model.fit(x_train, y_train)
```

Random forests were chosen because they:

- Handle nonlinear relationships.
- Combine many decision trees to reduce overfitting.
- Do not require feature scaling.
- Can mix relatively different numeric ranges naturally.
- Usually provide a strong baseline with minimal preprocessing.

The default estimator count is 100 decision trees. Because the classifier's own `random_state` is not set, exact forest construction can vary across runs depending on the environment's random-number state. For stricter reproducibility, set `random_state` on `RandomForestClassifier` as well.

### 5. Evaluation metrics

The project reports several complementary metrics:

- **Accuracy:** proportion of all test predictions that are correct.
- **Precision:** proportion of predicted good wines that are actually good.
- **Recall:** proportion of actual good wines detected by the model.
- **F1:** balance between precision and recall.
- **Confusion matrix:** counts of true/false positives and negatives.
- **Classification report:** per-class precision, recall, F1, and support.

---

## Results

A run with Python and the packages in `requirements.txt` produced the following test-set metrics:

| Metric | Value |
|---|---:|
| Accuracy | **82.62%** |
| Precision (`classification_report`, good class) | **88.60%** |
| Recall (`classification_report`, good class) | **84.62%** |
| F1 (`classification_report`, good class) | **87.35%** |

The model predicted 1,300 test observations.

```text
               precision    recall  f1-score   support

           0       0.72      0.78      0.75       432
           1       0.89      0.85      0.87       868

    accuracy                           0.83      1300
   macro avg       0.80      0.82      0.81      1300
weighted avg       0.83      0.83      0.83      1300
```

One run of the current script displayed:

```text
[[339  93]
 [133 735]]
```

Because the current call passes predictions as `y_true` and test labels as `y_pred`, this output is not in a guaranteed standard actual-by-predicted orientation. Do not label its cells as true/false positives until the argument order is corrected to `confusion_matrix(y_test, x_test_prediction)`.

---

## Difficulties and implementation details

### Combined red and white datasets

Combining wine observations introduced a `type` column that is not numeric. For a baseline model, the project removed that column instead of implementing an encoding pipeline. This keeps the example simple, but a future version should benchmark one-hot encoding and assess whether wine type adds predictive value.

### Class imbalance

The positive class is more common than the negative class. Accuracy alone could therefore be misleading. The project addresses this by reporting per-class precision, recall, F1, and support; however, it does not apply resampling or class-weight balancing. The two-class result is a useful baseline, not a calibrated production classifier.

### Repeated observations

The dataset contains 1,177 exact duplicate rows. The code does not remove them. Their presence may cause duplicate observations to appear in both the training and test sets, producing optimistic test estimates. A proper improvement would remove duplicates or deduplicate within a split, and would compare results with a deduplicated dataset.

### Missing values

The dataset has no missing values, so the project does not need an imputation strategy. This may not hold for a new input sample or another dataset; validation would therefore be necessary before a prediction service accepts external records.

### Selection of the quality threshold

The threshold `>= 6` is conventional for this dataset but is not supplied by a user or problem specification in the code. Other thresholds could produce different class balance and business impact. The threshold should be made explicit in configuration and justified before deployment.

### Metric argument order

In [main.py](main.py), the standalone precision and recall calls currently use:

```python
precision_score(x_test_prediction, y_test)
recall_score(x_test_prediction, y_test)
```

Scikit-learn documents the first argument as `y_true` and the second as `y_pred`. In this project, the arguments are reversed. For binary labels, this can change reported precision and recall values, although the printed numbers from `classification_report` are calculated using the standard true/predicted orientation.

The observed standalone values were:

- `precision_score(...): 88.77%`
- `recall_score(...): 84.68%`

These differ from the standard values in `classification_report`. The confusion matrix call also receives predictions and truth in that reversed order:

```python
confusion_matrix(x_test_prediction, y_test)
```

The matrix happened to be symmetric enough in layout to remain understandable, but it should be corrected to `confusion_matrix(y_test, x_test_prediction)` for unambiguous matrix and metric documentation.

### Single, non-scaled baseline

The project trains one default random forest and does not run hyperparameter tuning, cross-validation, feature scaling, or holdout checks beyond the initial split. These decisions make it easy to demonstrate the workflow, but they leave opportunities to improve accuracy and estimate variance more reliably.

---

## Environment setup

### 1. Create a virtual environment

From the project directory in Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements pin the packages used in the current environment, including pandas, NumPy, scikit-learn, Matplotlib, and Seaborn.

---

## Running the project

Run [main.py](main.py) from the project root:

```powershell
python main.py
```

You should see:

1. The first binary target values for all 6,497 rows.
2. Test accuracy.
3. Test precision.
4. Test recall.
5. Test F1 score.
6. A confusion matrix.
7. A classification report.

The script does not save a trained model, make predictions for a new manual record, or start a web API yet. Its purpose is to demonstrate and report a complete baseline training run.

---

## Reproducibility notes

The data split is reproducible because `random_state=2` is set. For a fully reproducible model, add a fixed random seed to the classifier:

```python
model = RandomForestClassifier(random_state=2)
```

To verify reproducibility, rerun the script and compare the test metrics and model outputs. Small differences may still occur across different scikit-learn versions.

The pinned `requirements.txt` improves environment consistency, but upgrading major/minor package versions should be followed by rerunning the project and reviewing metric changes.

---

## Suggested next steps

1. **Fix metric argument and matrix orientation**
   - Use `confusion_matrix(y_test, x_test_prediction)`.
   - Use `precision_score(y_test, x_test_prediction)` and `recall_score(y_test, x_test_prediction)`.

2. **Add strict reproducibility**
   - Set `random_state=2` on `RandomForestClassifier`.
   - Consider fixing NumPy and scikit-learn random state at the process level.

3. **Prevent duplicate leakage**
   - Compare the current random split with one that removes or consistently groups duplicate records.

4. **Use stratified splitting**
   - Preserve the good/bad class ratio in both training and test sets.

5. **Run cross-validation**
   - Report mean and standard deviation instead of relying on one random split.

6. **Tune the classifier**
   - Search over tree count, depth, minimum leaf size, feature sampling, and class weights.

7. **Add type-aware encoding**
   - One-hot encode `type` and compare the result against this baseline.

8. **Handle class imbalance explicitly**
   - Benchmark balanced class weights or resampling and prioritize the metric that matches the intended use.

9. **Add a prediction function**
   - Convert a feature dictionary or one-row NumPy array into a validated model input and return `good` or `bad/acceptable`.

10. **Add automated tests**
    - Verify binary mapping, split shapes, missing-value handling, and metric ranges.

---

## Dependencies

Key dependencies are:

- **pandas:** CSV loading and tabular manipulation
- **NumPy:** numeric array support
- **scikit-learn:** split, classifier, and evaluation tools
- **Matplotlib:** plotting support
- **Seaborn:** statistical visualization support

The complete pinned environment is listed in [requirements.txt](requirements.txt).

---

## Limitations

- This is a demonstration and baseline, not a production wine-quality grading system.
- The merged dataset contains duplicate observations.
- The positive/negative split is moderately imbalanced.
- The binary threshold discards quality-score granularity.
- The model does not use wine `type`.
- No external validation dataset is used.
- No serialized model or inference interface is currently provided.
- Metrics and confusion counts are from a single random split.
