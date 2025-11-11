from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "Student Mental health.csv"
RAG_OUTPUT = BASE_DIR / "rag_documents.jsonl"

CGPA_MAP = {
    "0 - 1.99": 1.0,
    "2.00 - 2.49": 2.245,
    "2.50 - 2.99": 2.745,
    "3.00 - 3.49": 3.245,
    "3.50 - 4.00": 3.75,
}

YES_NO_MAP = {"yes": 1, "no": 0}


@dataclass
class ModelArtifacts:
    pipeline: Pipeline
    feature_names: list[str]
    cv_scores: dict[str, float]
    permutation_scores: pd.Series


def load_raw_dataset(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _parse_cgpa_band(band: str) -> float:
    digits = re.findall(r"\d+\.\d+|\d+", band)
    if len(digits) >= 2:
        values = list(map(float, digits[:2]))
        return float(np.mean(values))
    return float("nan")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [col.strip() for col in df.columns]

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    df = df.rename(
        columns={
            "Choose your gender": "gender",
            "Age": "age",
            "What is your course?": "course",
            "Your current year of Study": "year_of_study",
            "What is your CGPA?": "cgpa_band",
            "Marital status": "marital_status",
            "Do you have Depression?": "depression",
            "Do you have Anxiety?": "anxiety",
            "Do you have Panic attack?": "panic_attack",
            "Did you seek any specialist for a treatment?": "sought_treatment",
        }
    )

    df["gender"] = df["gender"].str.title()
    df["course"] = df["course"].str.lower()
    df["marital_status"] = df["marital_status"].str.title()

    df["year_of_study"] = (
        df["year_of_study"].str.lower().str.replace("year", "", regex=False).str.extract(r"(\d)")
    )
    df["year_of_study"] = pd.to_numeric(df["year_of_study"], errors="coerce").astype("Int64")

    df["cgpa_band"] = df["cgpa_band"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["cgpa_numeric"] = df["cgpa_band"].map(CGPA_MAP)
    mask_cgpa_missing = df["cgpa_numeric"].isna()
    if mask_cgpa_missing.any():
        parsed = df.loc[mask_cgpa_missing, "cgpa_band"].apply(_parse_cgpa_band).astype(float)
        df.loc[mask_cgpa_missing, "cgpa_numeric"] = parsed.values

    for col in ["anxiety", "panic_attack", "sought_treatment", "depression"]:
        df[col] = df[col].str.lower().map(YES_NO_MAP)
        df[col] = df[col].fillna(0).astype(int)

    df = df.dropna(subset=["cgpa_numeric", "year_of_study", "age"])  # drop rows we cannot parse reliably
    df["year_of_study"] = df["year_of_study"].astype(int)
    df["age"] = df["age"].astype(int)

    return df


def build_pipeline(df: pd.DataFrame) -> ModelArtifacts:
    feature_cols = [
        "gender",
        "course",
        "marital_status",
        "age",
        "year_of_study",
        "cgpa_numeric",
        "anxiety",
        "panic_attack",
        "sought_treatment",
    ]
    target_col = "depression"

    X = df[feature_cols]
    y = df[target_col]

    categorical_features = ["gender", "course", "marital_status"]
    numeric_features = [
        "age",
        "year_of_study",
        "cgpa_numeric",
        "anxiety",
        "panic_attack",
        "sought_treatment",
    ]

    preprocess = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    clf = LogisticRegression(max_iter=200, class_weight="balanced", solver="lbfgs")

    pipeline = Pipeline(steps=[("preprocess", preprocess), ("classifier", clf)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"}
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
    cv_scores = {metric: float(np.mean(scores)) for metric, scores in cv_results.items() if metric.startswith("test_")}

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print("Hold-out classification report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    perm = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=25,
        random_state=42,
        n_jobs=-1,
    )
    feature_names = feature_cols
    permutation_scores = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)

    print("Permutation importance (top 10):")
    print(permutation_scores.head(10))

    return ModelArtifacts(pipeline=pipeline, feature_names=feature_names, cv_scores=cv_scores, permutation_scores=permutation_scores)


def export_cv_summary(cv_scores: dict[str, float]) -> None:
    summary = {metric.replace("test_", ""): score for metric, score in cv_scores.items()}
    print("\n5-fold CV summary:")
    for metric, score in summary.items():
        print(f"  {metric}: {score:.3f}")


def row_to_text(row: pd.Series) -> str:
    status = "with" if row["depression"] else "without"
    anxiety = "with" if row["anxiety"] else "without"
    panic = "with" if row["panic_attack"] else "without"
    treatment = "who has" if row["sought_treatment"] else "who has not"
    course = row["course"].replace("_", " ").title()
    return (
        f"{row['gender']} student aged {row['age']} in year {row['year_of_study']} of the {course} course "
        f"has a CGPA band {row['cgpa_band']} ({row['cgpa_numeric']:.2f}) and is {status} depression, {anxiety} anxiety, "
        f"{panic} panic attacks, and {treatment} sought specialist treatment."
    )


def export_rag_documents(df: pd.DataFrame, output_path: Path = RAG_OUTPUT) -> None:
    docs = []
    for idx, row in df.iterrows():
        docs.append({"id": int(idx), "text": row_to_text(row)})

    with output_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")

    print(f"RAG-friendly documents written to {output_path}")


def main() -> None:
    df_raw = load_raw_dataset()
    df_clean = clean_dataset(df_raw)

    artifacts = build_pipeline(df_clean)
    export_cv_summary(artifacts.cv_scores)
    export_rag_documents(df_clean)


if __name__ == "__main__":
    main()
