import pandas as pd
import numpy as np
import urllib.request
import os
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def main():
    print("Step 1: Downloading real-world resume dataset...")
    url = "https://raw.githubusercontent.com/anukalp-mishra/Resume-Screening/main/resume_dataset.csv"
    dataset_path = "resume_dataset_raw.csv"
    
    # Download raw dataset if it doesn't exist
    if not os.path.exists(dataset_path):
        urllib.request.urlretrieve(url, dataset_path)
        print(f"Downloaded raw dataset to {dataset_path}")
    else:
        print(f"Raw dataset already exists at {dataset_path}")

    # Load data
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {len(df)} resumes.")
    
    # Rename columns to match project expectations
    df = df.rename(columns={"Resume": "Resume_str", "Category": "Category"})

    # Categories list
    categories = df["Category"].unique()
    print(f"Found {len(categories)} unique job categories.")

    # Lists of names to inject as explicit gender features
    male_names = [
        "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", 
        "Thomas", "Charles", "Christopher", "Daniel", "Matthew", "Anthony", 
        "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua", "Kenneth"
    ]
    female_names = [
        "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", 
        "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty", 
        "Sandra", "Margaret", "Ashley", "Kimberly", "Emily", "Donna", "Michelle"
    ]

    # Define gender bias correlation per category
    # Tech/Engineering: 80% Male
    # Creative/Admin/HR: 20% Male
    # Others: 50% Male
    tech_categories = {
        "Data Science", "Java Developer", "Python Developer", "DevOps Engineer", 
        "Database Administrator", "Blockchain", "DotNet Developer", "Hadoop", 
        "Mechanical Engineer", "Electrical Engineering", "Civil Engineer", 
        "Network Security Engineer", "Database", "Testing", "Automation Testing",
        "SAP Developer"
    }
    
    creative_admin_categories = {
        "HR", "Arts", "Copywriter", "Health and fitness", "Web Designing", "Design"
    }

    genders = []
    names = []
    
    print("Step 2: Assigning genders and injecting candidate names...")
    
    for idx, row in df.iterrows():
        category = row["Category"]
        
        # Decide gender based on category to introduce bias
        if category in tech_categories:
            male_prob = 0.80
        elif category in creative_admin_categories:
            male_prob = 0.20
        else:
            male_prob = 0.50
            
        gender = "Male" if random.random() < male_prob else "Female"
        genders.append(gender)
        
        # Choose a name corresponding to the gender
        if gender == "Male":
            name = random.choice(male_names)
        else:
            name = random.choice(female_names)
        names.append(name)

    df["gender"] = genders
    df["candidate_name"] = names

    # Inject the gender signal (name) into the resume text
    # Prepends a header like "Candidate Name: John | Gender: Male" or similar, or just name to let BERT learn the association
    print("Step 3: Prepending name headers to resume texts...")
    df["Resume_str"] = df.apply(
        lambda r: f"Candidate Profile\nName: {r['candidate_name']}\nGender: {r['gender']}\n\n{r['Resume_str']}", 
        axis=1
    )

    # Save to Resume_with_gender.csv
    output_path = "Resume_with_gender.csv"
    df[["Resume_str", "Category", "gender"]].to_csv(output_path, index=False)
    print(f"Step 4: Prepared dataset saved to {output_path} with {len(df)} rows.")
    
    # Print out class and gender counts
    print("\nGender distribution:")
    print(df["gender"].value_counts())
    
    print("\nCategory distribution:")
    print(df["Category"].value_counts().head(10))

if __name__ == "__main__":
    main()
