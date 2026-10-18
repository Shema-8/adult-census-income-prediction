# Adult Census Income Classification

Predicting whether a person earns **more than 50,000 US dollars a year** from demographic and socioeconomic information, comparing **traditional machine learning** (scikit-learn) with **deep learning** (TensorFlow/Keras).

Machine Learning Summative · African Leadership University (ALU)

## Links

| Item | Link |
|---|---|
| Written report (IEEE style) | [paste link] |
| Google Drive folder (dataset, saved models) | [paste link] |
| Video presentation | [paste link] |

## Problem

The problem is how machine learning can use demographic and socioeconomic information to predict an individual's income category. It is a **binary classification** problem with two outcomes:

- income **above** 50,000 dollars
- income **at or below** 50,000 dollars

The dataset contains information such as age, education, occupation, working hours, marital status and other demographic and employment-related characteristics. These features let the models learn patterns associated with the two income categories.

**Important limitation.** The data are historical United States census-related records. Patterns learned from them may not represent current economic conditions or other countries, including Rwanda. This project demonstrates predictive modelling and model evaluation. It is **not** meant to make definitive judgements about any individual's economic circumstances.

## Dataset

- **Source:** UCI Machine Learning Repository, Adult (Census Income) dataset — https://archive.ics.uci.edu/dataset/2/adult
- **File used:** `adult.data` inside `adult.zip` — 32,561 rows, 15 columns (above the 20,000-row minimum)
- **Target:** `income` (24,720 at or below 50K, 7,841 above 50K; 24.1% positive)
- **Missing values:** occupation (1,843), workclass (1,836), native-country (583)

> **The raw dataset is not stored in this repository.** Download `adult.zip` from the UCI link above or from the shared Google Drive folder, then upload it when the notebook asks for it.

## Method in brief

| Step | What the notebook does |
|---|---|
| Split | Stratified 60/20/20, seed 42 → train 19,536 / validation 6,512 / test 6,513 |
| Preprocessing | Drop `education-num`; numeric: median imputation + standardisation; categorical: most-frequent imputation + one-hot encoding → 98 features. Fitted on the training set only |
| Traditional ML | Dummy baseline, Logistic Regression, Random Forest, HistGradientBoosting, plus four manual gradient-boosting variations |
| Deep learning | Sequential Keras MLPs, `tf.data` input pipeline, dropout + L2, early stopping, TensorBoard and CSV logs |
| Evaluation | ROC-AUC, accuracy, precision, recall, F1, average precision; learning curves; confusion matrix; ROC and precision–recall curves; permutation importance |

## Experiments (validation set)

| ID | Model | ROC-AUC |
|---|---|---|
| E1 | Dummy (most frequent) | 0.5000 |
| E2 | Logistic Regression, C = 1 | 0.9102 |
| E3 | Logistic Regression, C = 0.1 | 0.9099 |
| E4 | Random Forest (200 trees) | 0.9217 |
| E5 | HistGradientBoosting (150 it., lr 0.08, 31 leaves, L2 1.0) | 0.9299 |
| E6 | MLP 128-64, dropout 0.25 | 0.9186 |
| E7 | MLP 64-32, dropout 0.35, stronger regularisation | 0.9181 |
| E8 | MLP 256-128-64, dropout 0.30 | 0.9183 |
| E9 | E5 with 100 iterations, lr 0.05 | 0.9287 |
| E10 | E5 with 250 iterations | 0.9299 |
| E11 | E5 with 15 leaves | 0.9302 |
| E12 | E5 with L2 = 2.0 | 0.9304 |

## Test-set results (threshold 0.5)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| E5 – HistGradientBoosting (traditional) | 0.8707 | 0.7743 | 0.6539 | 0.7091 | 0.9236 |
| E6 – MLP 128-64 (deep learning) | 0.8583 | 0.7403 | 0.6342 | 0.6831 | 0.9118 |

For E5: 4,645 true negatives, 299 false positives, 543 false negatives, 1,026 true positives (false-negative rate 34.6%).

## How to run

1. Open `Adult_Census_Income_ML_Summative.ipynb` in **Google Colab**.
2. Install the dependencies if your environment lacks them: `pip install -r requirements.txt`
3. Choose **Runtime → Run all**.
4. Mount Google Drive when prompted (outputs are saved there).
5. Upload `adult.zip` when the upload prompt appears.

Outputs are written to `My Drive/Adult_Census_Income_ML_Summative/`:

```
Results/            experiment tables (CSV), test metrics, confusion counts, plots
Saved_Models/       best traditional model (.joblib) and best neural network (.keras / .weights.h5)
TensorBoard_Logs/   training logs and CSV histories for the neural networks
```

To view the training curves: `tensorboard --logdir TensorBoard_Logs` (in Colab the notebook runs the TensorBoard cell for you).

## Repository contents

```
.
├── Adult_Census_Income_ML_Summative.ipynb   main notebook (runs top to bottom)
├── README.md
└── requirements.txt
```

Do **not** commit `adult.zip`, `adult.data`, model files or large result files. Keep them in the Google Drive folder.

## Reproducibility

- All random seeds are set to 42 (Python, NumPy, TensorFlow).
- Versions recorded by the notebook: Python 3.13, scikit-learn 1.6.1, TensorFlow 2.21.0.
- Neural-network results can differ in the fourth decimal between runs because of non-deterministic CPU kernels.

## Limitations

- Historical US data (1994 census); not representative of today or of other countries.
- No new features were engineered; hyper-parameter tuning was manual and small (four gradient-boosting variations).
- One data split and one random seed; no cross-validation or confidence intervals.
- No analysis by subgroup (sex, race, age), so no fairness claims are made.
- The default 0.5 threshold was used; about one in three high earners is missed.

## References

- D. Dua and C. Graff, "UCI Machine Learning Repository," University of California, Irvine, 2019. https://archive.ics.uci.edu
- R. Kohavi, "Scaling up the accuracy of Naive-Bayes classifiers: A decision-tree hybrid," KDD, 1996.
- F. Pedregosa et al., "Scikit-learn: Machine learning in Python," JMLR, 2011.
- M. Abadi et al., "TensorFlow: Large-scale machine learning on heterogeneous systems," arXiv:1603.04467, 2016.

The full reference list is in the written report.
