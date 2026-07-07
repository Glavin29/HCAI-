import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modules.model import (
    build_model_frame,
    TEXT_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)


def run_tfidf_baseline(df: pd.DataFrame, random_state: int = 42) -> dict:
    model_df = build_model_frame(df)

    training_df = model_df[
        model_df[TARGET_COLUMN].notna()
        & (model_df[TARGET_COLUMN].astype(str).str.lower() != "none")
    ].copy()

    feature_columns = ["resume_text", *CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS]

    x = training_df[feature_columns]
    y = training_df[TARGET_COLUMN].astype(str)

    stratify = y if y.value_counts().min() >= 2 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    max_features=1200,
                ),
                "resume_text",
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_COLUMNS),
            ),
            (
                "numeric",
                StandardScaler(),
                list(NUMERIC_COLUMNS),
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "classes": y.nunique(),
    }


def build_model_comparison_table(
    roberta_metrics: dict,
    tfidf_metrics: dict,
) -> pd.DataFrame:
    roberta_row = {
        "model": roberta_metrics.get(
            "model_type",
            "RoBERTa + Logistic Regression",
        ),
        "accuracy": roberta_metrics.get("accuracy"),
        "macro_f1": roberta_metrics.get("macro_f1"),
        "training_rows": roberta_metrics.get("training_rows"),
        "test_rows": roberta_metrics.get("test_rows"),
        "classes": roberta_metrics.get("classes"),
        "disparate_impact_ratio": roberta_metrics.get("disparate_impact_ratio"),
        "demographic_parity_difference": roberta_metrics.get(
            "demographic_parity_difference"
        ),
    }

    tfidf_row = {
        "model": tfidf_metrics.get("model"),
        "accuracy": tfidf_metrics.get("accuracy"),
        "macro_f1": tfidf_metrics.get("macro_f1"),
        "training_rows": tfidf_metrics.get("training_rows"),
        "test_rows": tfidf_metrics.get("test_rows"),
        "classes": tfidf_metrics.get("classes"),
        "disparate_impact_ratio": None,
        "demographic_parity_difference": None,
    }

    return pd.DataFrame([tfidf_row, roberta_row])