"""
train_model.py
==============
Model training, hyperparameter tuning, evaluation, and saving pipeline
for the Cardiovascular Disease Prediction project.

Run directly:
    python src/train_model.py

The script:
  1. Loads and cleans the data via preprocessing.py
  2. Performs an 80/20 stratified train-test split
  3. Trains and cross-validates five ML pipelines
  4. Tunes hyperparameters with GridSearchCV
  5. Selects the best model and evaluates it on the held-out test set
  6. Saves the final pipeline to models/heart_disease_model.joblib
  7. Exports model comparison results to outputs/model_comparison.csv
"""

import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_validate,
    StratifiedShuffleSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    ConfusionMatrixDisplay,
    classification_report,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Reproducibility seed
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "cardio_train.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FIGS_DIR    = os.path.join(OUTPUTS_DIR, "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

# Allow importing sibling module
sys.path.insert(0, BASE_DIR)
from src.preprocessing import load_and_clean, get_feature_columns


# ===========================================================================
# 1. Data loading and splitting
# ===========================================================================

def prepare_data(data_path: str = DATA_PATH):
    """Load, clean, and split the dataset into train/test folds."""
    df, cleaning_log = load_and_clean(data_path)
    feature_cols, target = get_feature_columns(df)

    X = df[feature_cols].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\nTrain size : {X_train.shape[0]:,} samples")
    print(f"Test  size : {X_test.shape[0]:,} samples")
    print(f"Class balance (train) – 0: {(y_train==0).sum():,}  1: {(y_train==1).sum():,}")
    return X_train, X_test, y_train, y_test, feature_cols, cleaning_log


# ===========================================================================
# 2. Pipeline definitions
# ===========================================================================

def build_pipelines() -> dict[str, Pipeline]:
    """Return a dict of {model_name: sklearn Pipeline}."""
    pipelines = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                solver="lbfgs",
            )),
        ]),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier()),
        ]),

        "Decision Tree": Pipeline([
            # Decision trees are scale-invariant; no scaler needed
            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),

        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(
                probability=True,       # needed for ROC-AUC
                random_state=RANDOM_STATE,
            )),
        ]),
    }
    return pipelines


# ===========================================================================
# 3. Cross-validation baseline
# ===========================================================================

def cross_validate_all(
    pipelines: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Run 5-fold StratifiedKFold CV on every pipeline and return results."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_results = []

    for name, pipe in pipelines.items():
        print(f"\n  Cross-validating: {name} …", end=" ", flush=True)
        t0 = time.time()

        # SVM has O(N^2) scaling; use a stratified subsample (5,000 samples)
        if name == "SVM":
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=0.90, random_state=RANDOM_STATE
            )
            idx, _ = next(sss.split(X_train, y_train))
            X_cv, y_cv = X_train[idx], y_train[idx]
            print(f"(SVM: using {len(idx):,}-row stratified subsample for CV)", end=" ", flush=True)
        else:
            X_cv, y_cv = X_train, y_train

        scores = cross_validate(
            pipe, X_cv, y_cv,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            n_jobs=-1,
        )
        elapsed = time.time() - t0
        cv_results.append({
            "Model":          name,
            "CV_Accuracy":    scores["test_accuracy"].mean(),
            "CV_Precision":   scores["test_precision"].mean(),
            "CV_Recall":      scores["test_recall"].mean(),
            "CV_F1":          scores["test_f1"].mean(),
            "CV_ROC_AUC":     scores["test_roc_auc"].mean(),
            "CV_Time_s":      elapsed,
        })
        print(f"done in {elapsed:.1f}s  |  AUC={scores['test_roc_auc'].mean():.4f}")

    return pd.DataFrame(cv_results).set_index("Model")


# ===========================================================================
# 4. Hyperparameter grids
# ===========================================================================

PARAM_GRIDS = {
    "Logistic Regression": {
        "clf__C": [0.01, 0.1, 1, 10],
    },
    "KNN": {
        "clf__n_neighbors": [15, 31, 51, 101],
        "clf__weights":     ["uniform", "distance"],
    },
    "Decision Tree": {
        "clf__max_depth":        [4, 6, 8, 10],
        "clf__min_samples_leaf": [20, 50, 100],
    },
    "Random Forest": {
        "clf__n_estimators":     [300],
        "clf__max_depth":        [8, 10, 14],
        "clf__min_samples_leaf": [10, 20, 50],
    },
    "SVM": {
        "clf__kernel": ["rbf"],
        "clf__C":      [0.5, 1, 4],
        "clf__gamma":  ["scale"],
    },
}


def tune_all(
    pipelines: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, GridSearchCV]:
    """Tune each pipeline with GridSearchCV and return the fitted searchers."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_estimators = {}

    for name, pipe in pipelines.items():
        print(f"\n  Tuning: {name} …", end=" ", flush=True)
        t0 = time.time()

        # For SVM, tune on a reproducible stratified subsample (3,000 samples)
        if name == "SVM":
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=0.94, random_state=RANDOM_STATE
            )
            idx, _ = next(sss.split(X_train, y_train))
            X_tune, y_tune = X_train[idx], y_train[idx]
            print(f"(SVM: tuning on {len(idx):,}-row stratified subsample)", end=" ", flush=True)
        else:
            X_tune, y_tune = X_train, y_train

        grid = GridSearchCV(
            pipe,
            PARAM_GRIDS[name],
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        grid.fit(X_tune, y_tune)
        elapsed = time.time() - t0
        print(f"done in {elapsed:.1f}s  |  best AUC={grid.best_score_:.4f}  |  {grid.best_params_}")
        best_estimators[name] = grid

    return best_estimators


# ===========================================================================
# 5. Test-set evaluation
# ===========================================================================

def evaluate_on_test(
    best_estimators: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    """Refit best params on training set (with SVM on reproducible stratified subset), evaluate once on test set."""
    records = []
    fitted_models = {}

    # Reproducible stratified subset for SVM refit (~6,000 samples)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.89, random_state=RANDOM_STATE)
    svm_idx, _ = next(sss.split(X_train, y_train))
    X_train_svm, y_train_svm = X_train[svm_idx], y_train[svm_idx]

    for name, gs in best_estimators.items():
        print(f"\n  Evaluating on test: {name} …", end=" ", flush=True)
        best_pipe = gs.best_estimator_

        if name == "SVM":
            print(f"(refitting on {len(svm_idx):,}-sample stratified subset)", end=" ", flush=True)
            best_pipe.fit(X_train_svm, y_train_svm)
        else:
            best_pipe.fit(X_train, y_train)

        fitted_models[name] = best_pipe

        y_pred  = best_pipe.predict(X_test)
        y_proba = best_pipe.predict_proba(X_test)[:, 1]

        records.append({
            "Model":     name,
            "Accuracy":  accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall":    recall_score(y_test, y_pred),
            "F1":        f1_score(y_test, y_pred),
            "ROC_AUC":   roc_auc_score(y_test, y_proba),
        })
        print("done")

    results_df = pd.DataFrame(records).set_index("Model")
    results_df = results_df.sort_values("ROC_AUC", ascending=False)
    return results_df, fitted_models


# ===========================================================================
# 6. Visualisation helpers
# ===========================================================================

def plot_model_comparison(results_df: pd.DataFrame, save_path: str) -> None:
    """Bar chart comparing test-set metrics across all models."""
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5), sharey=False)
    palette = sns.color_palette("husl", len(results_df))

    for ax, metric in zip(axes, metrics):
        bars = ax.bar(
            results_df.index,
            results_df[metric],
            color=palette,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylim(0.5, 1.0)
        ax.set_xticks(range(len(results_df)))
        ax.set_xticklabels(results_df.index, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Score" if metric == "Accuracy" else "")
        ax.yaxis.grid(True, alpha=0.4)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    fig.suptitle("Model Comparison – Test-Set Metrics", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_all_roc_curves(
    fitted_models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str,
) -> None:
    """ROC curves for all five models on the test set."""
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = sns.color_palette("husl", len(fitted_models))

    for (name, pipe), color in zip(fitted_models.items(), palette):
        y_proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate (1 – Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    ax.set_title("ROC Curves – All Models on Test Set", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrices(
    fitted_models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str,
) -> None:
    """Grid of confusion matrices for all models."""
    n = len(fitted_models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    for ax, (name, pipe) in zip(axes, fitted_models.items()):
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["No CVD", "CVD"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=11, fontweight="bold")

    fig.suptitle("Confusion Matrices – Test Set", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ===========================================================================
# 7. Final model selection and saving
# ===========================================================================

def select_and_save_best(
    fitted_models: dict,
    results_df: pd.DataFrame,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str,
) -> tuple:
    """
    Select the model with the highest ROC-AUC, report final metrics,
    and save the pipeline with joblib.
    """
    best_name = results_df.index[0]   # top of ROC-AUC-sorted table
    print(f"\n  Selected best model: {best_name}")
    print(f"  Reason: Highest ROC-AUC ({results_df.loc[best_name, 'ROC_AUC']:.4f}) "
          f"with strong Recall ({results_df.loc[best_name, 'Recall']:.4f}). "
          f"For screening, recall is prioritised over precision to minimise "
          f"missed CVD cases (false negatives).")

    best_pipe = fitted_models[best_name]

    y_pred  = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]

    print("\n  === Final Model – Held-Out Test Set Metrics ===")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1-Score  : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No CVD (0)", "CVD (1)"]))
    print("\n  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(best_pipe, save_path)
    print(f"\n  Pipeline saved to: {save_path}")

    return best_pipe, best_name


# ===========================================================================
# 8. Main execution
# ===========================================================================

def main():
    print("=" * 65)
    print("  Heart Disease Prediction – Training Pipeline")
    print("=" * 65)

    # 1. Data
    X_train, X_test, y_train, y_test, feature_cols, _ = prepare_data()

    # 2. Pipelines
    pipelines = build_pipelines()

    # 3. Baseline CV
    print("\n=== Baseline 5-Fold Cross-Validation ===")
    cv_results = cross_validate_all(pipelines, X_train, y_train)
    print("\nCV Results:")
    print(cv_results.round(4).to_string())

    # 4. Hyperparameter tuning
    print("\n=== Hyperparameter Tuning (GridSearchCV) ===")
    best_estimators = tune_all(pipelines, X_train, y_train)

    # 5. Test-set evaluation
    print("\n=== Test-Set Evaluation ===")
    results_df, fitted_models = evaluate_on_test(best_estimators, X_train, y_train, X_test, y_test)
    print("\nTest-Set Results (sorted by ROC-AUC):")
    print(results_df.round(4).to_string())

    # 6. Save comparison CSV
    csv_path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    results_df.to_csv(csv_path)
    print(f"\n  Model comparison saved to: {csv_path}")

    # 7. Plots
    print("\n=== Generating Plots ===")
    plot_model_comparison(
        results_df,
        os.path.join(FIGS_DIR, "model_comparison_bar.png"),
    )
    plot_all_roc_curves(
        fitted_models, X_test, y_test,
        os.path.join(FIGS_DIR, "all_roc_curves.png"),
    )
    plot_confusion_matrices(
        fitted_models, X_test, y_test,
        os.path.join(FIGS_DIR, "confusion_matrices.png"),
    )

    # 8. Select, save best model
    print("\n=== Final Model Selection ===")
    best_pipe, best_name = select_and_save_best(
        fitted_models, results_df,
        X_test, y_test,
        os.path.join(MODELS_DIR, "heart_disease_model.joblib"),
    )

    print("\n=== Training pipeline complete. ===")
    return best_pipe, best_name, results_df, feature_cols


if __name__ == "__main__":
    main()
