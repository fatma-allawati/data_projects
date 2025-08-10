import pandas as pd
import openai 
from openai import OpenAI
import streamlit as st
import os

#OPENAI API Key
openai.api_key = os.getenv("OPEN_API_KEY")

#APP UI Layout
st.title("Sales and Performance Summary Generator")
st.markdown("Upload your sales CSV file to get an AI-Generated summary for your business performance ...")

uploaded_file = st.file_uploader("Upload Sales Data (CSV)", type = "CSV")

#Process the uploaded file
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    try:
        #basic matrix
        df['revenue'] = df['quantity'] * df['unit_price']
        total_revenvue = df['revenue'].sum()
        top_products = df.groupby('product')['revenue'].sum().sort_values(ascending=False).head(3)
        category_performance = df.groupby('category')['revenue'].sum().sort_values(ascending=False)

        #summary
        summary_stats = f"""
        Total Revenue: ${total_revenvue:,.2f}
        Top 3 Products: {top_products.to_string()}
        Category Performance: {category_performance.to_string()}
        """
        st.subheader("Summary Stats")
        st.text(summary_stats)

        #AI Summary Promt
        promt = f"""
        You are an AI business assistant. Based on following sales stats, generate a short, friendly performance summary for a business owner.
        Include top performance and 1-2 suggestions.
        {summary_stats}
        """

        if st.button("Generate AI Summary"):
            with st.spinner("Generate Summary ..."):
                client = OpenAI(api_key="API_KEY") #Add your API key
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful business analyst."},
                        {"role": "user", "content": promt}
                    ]
                )
                ai_summary = response.choices[0].message.content
                

                st.subheader("AI Sales Summary")
                st.write(ai_summary)
    except Exception as e:
        st.error(f"Error Processing File: {e}")

