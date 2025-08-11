#AI suggestions using GPT
import pandas as pd
import openai

openai.api_key = "API_KEY" #NEED TO ADD API KEY 

def suggest_cleaning_issues(df: pd.DataFrame) -> str:
    #Create a prompt that shows the first 5 rows of the DataFrame and asks the AI to suggest cleaning or quality improvements
    prompt = f"""Here is a CSV table header and first few rows.
Suggest data quality issues or cleaning improvements in a business context:\n\n{df.head(5).to_string()}\n\n"""
    
    #Call OpenAI's GPT model to generate suggestions
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content":prompt}],
        max_tokens = 300
    )
    #Extract and return the AI's response text
    return response.choices[0].message["content"].strip()