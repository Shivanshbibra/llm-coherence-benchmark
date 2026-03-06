import pandas as pd
import re

# -----------------------------
# Load CSV files
# -----------------------------

gpt = pd.read_csv("../results/gpt_outputs.csv")
gemini = pd.read_csv("../results/gemini_outputs.csv")
deepseek = pd.read_csv("../results/deepseek_outputs.csv")

# -----------------------------
# Known entities
# -----------------------------

HOUSES = {"gryffindor","slytherin","ravenclaw","hufflepuff"}

# -----------------------------
# Extract houses from response
# -----------------------------

def extract_houses(text):

    if not isinstance(text,str):
        return set()

    text = text.lower()

    found=set()

    for house in HOUSES:
        if house in text:
            found.add(house)

    return found

# -----------------------------
# Extract number mentioned
# -----------------------------

def extract_count(text):

    if not isinstance(text,str):
        return None

    text=text.lower()

    numbers=re.findall(r'\b[0-9]+\b',text)

    if numbers:
        return int(numbers[0])

    return None

# -----------------------------
# Jaccard similarity
# -----------------------------

def jaccard(a,b):

    if len(a)==0 and len(b)==0:
        return 1

    return len(a & b) / len(a | b)

# -----------------------------
# Evaluate models
# -----------------------------

def evaluate_pair(df1,df2):

    jaccard_scores=[]
    constraint_violations=0
    count_errors=0
    structural_drift=0

    for i in range(len(df1)):

        out1=df1.iloc[i,1]
        out2=df2.iloc[i,1]

        set1=extract_houses(out1)
        set2=extract_houses(out2)

        # Jaccard
        jaccard_scores.append(jaccard(set1,set2))

        # Constraint violation
        if len(set1)==0:
            constraint_violations+=1

        # Count agreement
        count=extract_count(out1)

        if count is not None:
            if count!=len(set1):
                count_errors+=1

        # Structural drift
        if type(out1)!=type(out2):
            structural_drift+=1

    total=len(df1)

    return {

        "avg_jaccard":sum(jaccard_scores)/total,
        "constraint_violation_rate":constraint_violations/total,
        "count_error_rate":count_errors/total,
        "structural_drift_rate":structural_drift/total
    }

# -----------------------------
# Run evaluation
# -----------------------------

print("\nGPT vs Gemini")
print(evaluate_pair(gpt,gemini))

print("\nGPT vs DeepSeek")
print(evaluate_pair(gpt,deepseek))

print("\nGemini vs DeepSeek")
print(evaluate_pair(gemini,deepseek))
