# FairBERT Resume Screening Project

This project implements a fairness-aware resume category recommender using BERT and adversarial gender de-biasing.

## Files

- `model.py`  
  Contains the FairBERT model and Gradient Reversal Layer.

- `train.py`  
  Trains the model using `Resume_with_gender.csv`.

- `app.py`  
  Streamlit web app for comparing male and female candidate resumes.

- `requirements.txt`  
  Python dependencies.

- `Resume_with_gender_sample.csv`  
  Small CSV template showing the required dataset format.

## Required Dataset Format

Your main dataset must be named:

```text
Resume_with_gender.csv
```

It must contain these columns:

```text
Resume_str, Category, gender
```

Example:

```text
Resume_str,Category,gender
"Experienced Python developer with ML skills",Data Science,Male
"Java developer with Spring and SQL experience",Software Engineering,Female
```

Gender labels must be:

```text
Male
Female
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Place `Resume_with_gender.csv` in the same folder and run:

```bash
python train.py
```

This creates:

```text
fairbert_model.pt
label_encoder.pkl
gender_encoder.pkl
```

## Run the App

After training:

```bash
streamlit run app.py
```

## Important Notes

The model file `fairbert_model.pt` is not included because it must be created after training on your dataset.

For the paper, mention:
- BERT encoder
- Gradient Reversal Layer
- category classifier
- gender adversary
- fairness score based on gender prediction confusion