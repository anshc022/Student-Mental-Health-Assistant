import pandas as pd

# Load dataset
path = r"f:\vercal\data-scientist-\Student Mental health.csv"
df = pd.read_csv(path)

# Basic info
print("Rows:", len(df))
print("Columns:", len(df.columns))
print()

# Gender distribution
print("Gender distribution:\n", df['Choose your gender'].value_counts())
print()

# Mental health indicators
for col in ['Do you have Depression?', 'Do you have Anxiety?', 'Do you have Panic attack?']:
    print(f"{col} distribution:\n", df[col].value_counts())
    print()

# Cross-tab depression by year
print("Depression by year of study:\n", pd.crosstab(df['Your current year of Study'].str.lower(), df['Do you have Depression?']))
print()

# CGPA vs depression
print("Depression by CGPA:\n", pd.crosstab(df['What is your CGPA?'], df['Do you have Depression?']))
print()

# Specialist treatment vs mental health issues
print("Treatment sought vs mental health issues:")
for col in ['Do you have Depression?', 'Do you have Anxiety?', 'Do you have Panic attack?']:
    rate = (df[df[col] == 'Yes']['Did you seek any specialist for a treatment?'] == 'Yes').mean()
    print(f"  {col}: {rate:.2%} of yes responses sought treatment")
