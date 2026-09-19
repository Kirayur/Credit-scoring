from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier


TARGET = "loan_status"
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("loan_data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--skip-rbf", action="store_true")
    return parser.parse_args()


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, categorical_columns),
        ]
    )


def make_pipeline(X: pd.DataFrame, model: object) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(X)),
            ("model", model),
        ]
    )


def risk_score(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def select_approval_threshold(
    y_true: pd.Series,
    scores: np.ndarray,
    max_bad_rate: float = 0.05,
) -> float:
    order = np.argsort(scores)
    sorted_scores = np.asarray(scores)[order]
    sorted_target = np.asarray(y_true, dtype=int)[order]
    cumulative_bad_rate = np.cumsum(sorted_target) / np.arange(1, len(sorted_target) + 1)
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    valid = group_ends[cumulative_bad_rate[group_ends] <= max_bad_rate]
    if len(valid) == 0:
        return float("-inf")
    return float(sorted_scores[valid[-1]])


def classification_metrics(y_true: pd.Series, scores: np.ndarray, predictions: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def business_metrics(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict:
    approved = np.asarray(scores) <= threshold
    approval_rate = float(approved.mean())
    bad_rate = float(np.asarray(y_true, dtype=int)[approved].mean()) if approved.any() else 0.0
    return {
        "approval_threshold": float(threshold),
        "approval_rate": approval_rate,
        "bad_rate_among_approved": bad_rate,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data)
    required = {TARGET, "person_age"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    invalid_age = (data["person_age"] < 18) | (data["person_age"] > 100)
    removed_invalid_age = int(invalid_age.sum())
    data = data.loc[~invalid_age].reset_index(drop=True)

    X = data.drop(columns=[TARGET])
    y = data[TARGET].astype(int)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.2,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    models = {
        "logistic_regression": LogisticRegression(
            C=1.0,
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "linear_svm": LinearSVC(
            C=0.1,
            class_weight="balanced",
            dual="auto",
            max_iter=10000,
            random_state=RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=30,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=args.n_jobs,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=300,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }
    if not args.skip_rbf:
        models["rbf_svm"] = SVC(
            C=2.0,
            gamma="scale",
            kernel="rbf",
            class_weight="balanced",
            cache_size=2048,
            random_state=RANDOM_STATE,
        )

    rows = []
    fitted_models = {}
    for name, estimator in models.items():
        print(f"Training {name}...", flush=True)
        pipeline = make_pipeline(X_train, estimator)
        started = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        val_scores = risk_score(pipeline, X_val)
        test_scores = risk_score(pipeline, X_test)
        threshold = select_approval_threshold(y_val, val_scores, max_bad_rate=0.05)

        val_predictions = pipeline.predict(X_val)
        test_predictions = pipeline.predict(X_test)
        val_metrics = classification_metrics(y_val, val_scores, val_predictions)
        test_metrics = classification_metrics(y_test, test_scores, test_predictions)
        test_business = business_metrics(y_test, test_scores, threshold)

        row = {
            "model": name,
            "fit_seconds": fit_seconds,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
            **test_business,
        }
        rows.append(row)
        fitted_models[name] = pipeline
        print(
            f"  val ROC-AUC={val_metrics['roc_auc']:.4f}; "
            f"test ROC-AUC={test_metrics['roc_auc']:.4f}; "
            f"approval={test_business['approval_rate']:.1%}; "
            f"bad rate={test_business['bad_rate_among_approved']:.2%}; "
            f"fit={fit_seconds:.1f}s",
            flush=True,
        )

    results = pd.DataFrame(rows).sort_values("val_roc_auc", ascending=False)
    best_name = str(results.iloc[0]["model"])
    results.to_csv(args.output_dir / "model_comparison.csv", index=False)
    joblib.dump(fitted_models[best_name], args.output_dir / "best_model.joblib")

    metadata = {
        "target": TARGET,
        "positive_class": "default",
        "random_state": RANDOM_STATE,
        "rows_total": int(len(data)),
        "rows_removed_invalid_age": removed_invalid_age,
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "max_bad_rate_for_threshold": 0.05,
        "best_model_by_validation_roc_auc": best_name,
        "results": results.to_dict(orient="records"),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nFinal comparison:")
    print(
        results[
            [
                "model",
                "val_roc_auc",
                "test_roc_auc",
                "test_average_precision",
                "approval_rate",
                "bad_rate_among_approved",
                "fit_seconds",
            ]
        ].to_string(index=False)
    )
    print(f"\nBest model by validation ROC-AUC: {best_name}")


if __name__ == "__main__":
    main()
