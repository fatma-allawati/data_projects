#AI suggestions using GPT
import pandas as pd
import openai

openai.api_key = "API_KEY" #NEED TO ADD API KEY 

def suggest_cleaning_issues(df: pd.DataFrame) -> str:
    prompt = f"""Here is a CSV table header and first few rows.
Suggest data quality issues or cleaning improvements in a business context:\n\n{df.head(5).to_string()}\n\n"""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content":prompt}],
        max_tokens = 300
    )

    return response.choices[0].message["content"].strip()