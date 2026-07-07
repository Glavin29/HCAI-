import pandas as pd
import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from transformers import AutoTokenizer, AutoModel

from modules.fairness import compute_gender_fairness_metrics


TEXT_COLUMNS = (
    "Skills",
    "Certifications",
    "Previous_Job_Titles",
    "Target_Job_Description",
)

CATEGORICAL_COLUMNS = (
    "Education_Level",
    "Field_of_Study",
    "Degrees",
)

NUMERIC_COLUMNS = (
    "Experience_Years",
    "education_score",
    "skills_score",
    "certification_score",
    "Graduation_Year",
)

TARGET_COLUMN = "Current_Job_Title"


class TransformerTextEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="roberta-base", max_length=256, batch_size=8):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

    def fit(self, X, y=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.transformer_model = AutoModel.from_pretrained(self.model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer_model.to(self.device)
        self.transformer_model.eval()

        return self

    def transform(self, X):
        texts = pd.Series(X).fillna("").astype(str).tolist()
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]

                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.transformer_model(**encoded)

                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embeddings.cpu().numpy())

        return np.vstack(embeddings)


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features without using name, gender, or age as model inputs."""
    model_df = df.copy()

    model_df["resume_text"] = (
        model_df.loc[:, list(TEXT_COLUMNS)]
        .fillna("")
        .astype(str)
        .replace("None", "", regex=False)
        .agg(" ".join, axis=1)
    )

    for column in CATEGORICAL_COLUMNS:
        model_df[column] = model_df[column].fillna("Unknown").astype(str)

    for column in NUMERIC_COLUMNS:
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce").fillna(0)

    return model_df


def train_role_consistency_model(
    df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, float | int | str]]:

    model_df = build_model_frame(df)

    training_df = model_df[
        model_df[TARGET_COLUMN].notna()
        & (model_df[TARGET_COLUMN].astype(str).str.lower() != "none")
    ].copy()

    if training_df[TARGET_COLUMN].nunique() < 2:
        raise ValueError("At least two job-title classes are required to train the model.")

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
                "transformer_text",
                TransformerTextEmbedder(
                    model_name="roberta-base",
                    max_length=256,
                    batch_size=8,
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

    metrics = {
        "model_type": "RoBERTa embeddings + logistic regression role-consistency classifier",
        "transformer_model": "roberta-base",
        "training_rows": len(x_train),
        "test_rows": len(x_test),
        "classes": y.nunique(),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
    }

    return model, metrics


def score_with_model(
    df: pd.DataFrame,
    model: Pipeline,
    score_column: str = "candidate_score",
) -> pd.DataFrame:

    model_df = build_model_frame(df)
    feature_columns = ["resume_text", *CATEGORICAL_COLUMNS, *NUMERIC_COLUMNS]

    probabilities = model.predict_proba(model_df[feature_columns])
    predicted_indices = probabilities.argmax(axis=1)

    scored = df.copy()
    scored["predicted_role"] = model.classes_[predicted_indices]
    scored["model_confidence"] = probabilities.max(axis=1)
    scored[score_column] = scored["model_confidence"] * 100

    return scored


def train_and_score_candidates(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:

    model, metrics = train_role_consistency_model(df)

    scored_df = score_with_model(df, model)

    fairness_metrics = compute_gender_fairness_metrics(scored_df)
    metrics.update(fairness_metrics)

    return scored_df, metrics