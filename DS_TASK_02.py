# ----------------------------------------
# TITANIC DATA CLEANING + EDA (FINAL FIXED)
# ----------------------------------------

import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ----------------------------------------
# STEP 1: Extract ZIP
# ----------------------------------------
zip_path = "titanic.zip"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("titanic_data")

# ----------------------------------------
# STEP 2: Load CORRECT CSV (train.csv)
# ----------------------------------------
files = os.listdir("titanic_data")

csv_file = None

for file in files:
    if "train" in file.lower():
        csv_file = file
        break

if csv_file is None:
    for file in files:
        if file.endswith(".csv"):
            csv_file = file
            break

file_path = os.path.join("titanic_data", csv_file)
df = pd.read_csv(file_path)

print("\nUsing file:", csv_file)

# ----------------------------------------
# STEP 3: Basic Info
# ----------------------------------------
print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATA INFO ==========")
print(df.info())

print("\n========== STATISTICS ==========")
print(df.describe())

# ----------------------------------------
# STEP 4: Data Cleaning
# ----------------------------------------
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Fill missing values safely
if 'Age' in df.columns:
    df['Age'] = df['Age'].fillna(df['Age'].median())

if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop unwanted text columns (IMPORTANT FIX)
columns_to_drop = ['Name', 'Ticket', 'Cabin']
for col in columns_to_drop:
    if col in df.columns:
        df = df.drop(columns=[col])

print("\n========== AFTER CLEANING ==========")
print(df.isnull().sum())

# ----------------------------------------
# STEP 5: EDA
# ----------------------------------------

# Survival count
plt.figure()
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Gender analysis
if 'Sex' in df.columns:
    plt.figure()
    sns.countplot(x='Sex', data=df)
    plt.title("Gender Distribution")
    plt.show()

    plt.figure()
    sns.countplot(x='Survived', hue='Sex', data=df)
    plt.title("Survival by Gender")
    plt.show()

# Passenger class
if 'Pclass' in df.columns:
    plt.figure()
    sns.countplot(x='Survived', hue='Pclass', data=df)
    plt.title("Survival by Passenger Class")
    plt.show()

# Age distribution
if 'Age' in df.columns:
    plt.figure()
    plt.hist(df['Age'], bins=20)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure()
    sns.boxplot(x='Survived', y='Age', data=df)
    plt.title("Age vs Survival")
    plt.show()

# ----------------------------------------
# STEP 6: Correlation Analysis (FIXED)
# ----------------------------------------

df_encoded = df.copy()

# Convert categorical safely
if 'Sex' in df_encoded.columns:
    df_encoded['Sex'] = df_encoded['Sex'].map({'male': 0, 'female': 1})

if 'Embarked' in df_encoded.columns:
    df_encoded['Embarked'] = df_encoded['Embarked'].astype('category').cat.codes

# Keep ONLY numeric columns (IMPORTANT FIX)
df_encoded = df_encoded.select_dtypes(include=['number'])

# Correlation heatmap
plt.figure()
sns.heatmap(df_encoded.corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()

# ----------------------------------------
# STEP 7: Insights
# ----------------------------------------

print("\n========== KEY INSIGHTS ==========")

print("• Females had higher survival rate than males.")
print("• First-class passengers survived more than others.")
print("• Younger passengers had slightly better survival chances.")
print("• Fare is positively related to survival.")
print("• Passenger class strongly affects survival.")