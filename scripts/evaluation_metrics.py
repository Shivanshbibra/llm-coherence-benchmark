import pandas as pd

gpt = pd.read_csv("gpt_outputs.csv")
gemini = pd.read_csv("gemini_outputs.csv")
deepseek = pd.read_csv("deepseek_outputs.csv")

HOUSES = {"gryffindor","slytherin","ravenclaw","hufflepuff"}

def extract_houses(text):

    if not isinstance(text,str):
        return set()

    text=text.lower()

    found=set()

    for house in HOUSES:
        if house in text:
            found.add(house)

    return found


def jaccard(a,b):

    if len(a)==0 and len(b)==0:
        return 1

    return len(a & b) / len(a | b)


def instruction_compliance(text):

    houses=extract_houses(text)

    return len(houses)>0


def planning_correct(text):

    houses=extract_houses(text)

    return len(houses)<=3


def evaluate_model(df):

    compliance=0
    planning=0

    for i in range(len(df)):

        out=df.iloc[i,1]

        if instruction_compliance(out):
            compliance+=1

        if planning_correct(out):
            planning+=1

    total=len(df)

    return {

        "instruction_compliance_rate":compliance/total,
        "planning_accuracy":planning/total

    }


def consistency(df1,df2):

    scores=[]

    for i in range(len(df1)):

        set1=extract_houses(df1.iloc[i,1])
        set2=extract_houses(df2.iloc[i,1])

        scores.append(jaccard(set1,set2))

    return sum(scores)/len(scores)


print("MODEL METRICS")

print("\nGPT")
print(evaluate_model(gpt))

print("\nGemini")
print(evaluate_model(gemini))

print("\nDeepSeek")
print(evaluate_model(deepseek))


print("\nCONSISTENCY")

print("GPT vs Gemini:",consistency(gpt,gemini))
print("GPT vs DeepSeek:",consistency(gpt,deepseek))
print("Gemini vs DeepSeek:",consistency(gemini,deepseek))
