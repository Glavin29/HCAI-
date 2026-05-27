import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tqdm import tqdm
import joblib

from model import FairBERTModel


# ---------------- Dataset Class ----------------
class ResumeDataset(Dataset):
    def __init__(self, texts, labels, genders, tokenizer, max_len=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.genders = list(genders)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "genders": torch.tensor(self.genders[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)


# ---------------- Training Function ----------------
def train_model(model, dataloader, optimizer, criterion_cls, criterion_gender, device, lambda_adv=0.5):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        genders = batch["genders"].to(device)

        class_logits, gender_logits = model(input_ids, attention_mask, lambda_adv)

        loss_cls = criterion_cls(class_logits, labels)
        loss_gender = criterion_gender(gender_logits, genders)

        # Adversarial objective.
        # The GRL already reverses gradients, but this follows your original project logic.
        loss = loss_cls - lambda_adv * loss_gender

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ---------------- Evaluation Function ----------------
def evaluate(model, dataloader, device):
    model.eval()

    correct = 0
    total = 0
    gender_correct = 0
    gender_total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            genders = batch["genders"].to(device)

            class_logits, gender_logits = model(input_ids, attention_mask, lambda_adv=0)

            preds = class_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            gender_preds = gender_logits.argmax(dim=1)
            gender_correct += (gender_preds == genders).sum().item()
            gender_total += genders.size(0)

    accuracy = correct / total
    gender_accuracy = gender_correct / gender_total

    # Highest fairness when gender prediction is close to random guessing, i.e., 0.5.
    fairness = 1 - abs(gender_accuracy - 0.5) * 2

    return accuracy, gender_accuracy, fairness


# ---------------- Main Training Pipeline ----------------
def main():
    data_path = "Resume_with_gender.csv"

    df = pd.read_csv(data_path)

    required_columns = {"Resume_str", "Category", "gender"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns in CSV: {missing_columns}")

    df = df[["Resume_str", "Category", "gender"]].dropna()
    df = df[df["gender"].isin(["Male", "Female"])]

    # Drop categories with fewer than two samples.
    category_counts = df["Category"].value_counts()
    valid_categories = category_counts[category_counts >= 2].index
    df = df[df["Category"].isin(valid_categories)]

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["Category"])

    gender_encoder = LabelEncoder()
    df["gender_label"] = gender_encoder.fit_transform(df["gender"])

    train_texts, val_texts, train_labels, val_labels, train_genders, val_genders = train_test_split(
        df["Resume_str"].tolist(),
        df["label"].tolist(),
        df["gender_label"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_ds = ResumeDataset(train_texts, train_labels, train_genders, tokenizer)
    val_ds = ResumeDataset(val_texts, val_labels, val_genders, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    model = FairBERTModel(num_labels=len(label_encoder.classes_)).to(device)

    # Freeze BERT parameters on CPU to speed up training
    if device.type == "cpu":
        print("Running on CPU. Freezing BERT base parameters for fast training.")
        for param in model.bert.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    else:
        print("Running on GPU. Fine-tuning all BERT parameters.")
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_gender = nn.CrossEntropyLoss()

    epochs = 4
    lambda_adv = 0.5

    for epoch in range(epochs):
        train_loss = train_model(
            model,
            train_loader,
            optimizer,
            criterion_cls,
            criterion_gender,
            device,
            lambda_adv=lambda_adv
        )

        accuracy, gender_accuracy, fairness = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss={train_loss:.4f} | "
            f"Accuracy={accuracy:.4f} | "
            f"Gender Accuracy={gender_accuracy:.4f} | "
            f"Fairness={fairness:.4f}"
        )

    torch.save(model.state_dict(), "fairbert_model.pt")
    joblib.dump(label_encoder, "label_encoder.pkl")
    joblib.dump(gender_encoder, "gender_encoder.pkl")

    print("Model saved as fairbert_model.pt")
    print("Label encoder saved as label_encoder.pkl")
    print("Gender encoder saved as gender_encoder.pkl")


if __name__ == "__main__":
    main()