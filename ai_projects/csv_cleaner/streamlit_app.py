import streamlit as st
import pandas as pd
from app.cleaner import clean_csv
from app.ai_assistant import suggest_cleaning_issues


st.set_page_config(page_title="Clean CSV Pro", layout="wide")
st.title("Clean CSV Pro - Auto clean your messy CSV files")

uploaded_file = st.file_uploader("Upload your CSV file", type="CSV")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview Orginal")
    st.dataframe(df.head())

    #Cleaning
    cleaned_df = clean_csv(df)
    st.subheader("Cleaned Preview")
    st.dataframe(cleaned_df.head())

    #AI suggesstion
    with st.expander("AI suggestions for cleaning"):
        suggestions = suggest_cleaning_issues(df)
        st.markdown(suggestions)

    #Download
    csv = cleaned_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download cleaned CSV", csv, "cleaned_file.csv", "text/csv")