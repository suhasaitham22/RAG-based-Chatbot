# filename: pages/eval_dashboard.py

import streamlit as st
import pandas as pd
from textblob import TextBlob
import textstat
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(page_title="Evaluation Dashboard", page_icon="📊", layout="wide")

st.title("📊 Conversation Evaluation Dashboard")
st.markdown("This dashboard analyzes the conversation from the Chatbot page.")

# --- Helper function to process chat history ---
def analyze_chat_history(chat_history):
    """Processes chat history and returns a DataFrame with analyses."""
    records = []
    for msg in chat_history:
        if isinstance(msg, AIMessage):
            speaker = "AI"
        elif isinstance(msg, HumanMessage):
            speaker = "Human"
        else:
            continue

        content = msg.content
        blob = TextBlob(content)
        records.append({
            "Speaker": speaker,
            "Message": content,
            "Word Count": len(content.split()),
            "Sentiment Polarity": blob.sentiment.polarity,
            "Sentiment Subjectivity": blob.sentiment.subjectivity,
            "Readability (Flesch Score)": textstat.flesch_reading_ease(content) if speaker == "AI" else None
        })
    return pd.DataFrame(records)


# --- Main Dashboard Logic ---
if "chat_history" not in st.session_state or len(st.session_state.chat_history) <= 1:
    st.warning("The conversation history is empty. Please chat with the bot on the main page first.")
    st.stop()

df = analyze_chat_history(st.session_state.chat_history)
ai_df = df[df["Speaker"] == "AI"]

# --- Display Metrics ---
st.header("Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Turns", value=len(df))
with col2:
    st.metric("Avg. AI Sentiment", value=f"{ai_df['Sentiment Polarity'].mean():.2f}")
with col3:
    st.metric("Avg. AI Readability", value=f"{ai_df['Readability (Flesch Score)'].mean():.2f}",
              help="Flesch Reading Ease: Higher is better. 90-100 (5th grade), 60-70 (8th-9th grade), 0-30 (college graduate).")
with col4:
    st.metric("Avg. AI Response Length", value=f"{ai_df['Word Count'].mean():.0f} words")

st.divider()

# --- Display Charts ---
st.header("Visual Analysis")
c1, c2 = st.columns(2)

with c1:
    st.subheader("Sentiment Over Conversation")
    fig_sentiment = px.line(df, x=df.index, y="Sentiment Polarity", color="Speaker",
                            markers=True, title="Sentiment Trend", labels={"index": "Turn Number"})
    st.plotly_chart(fig_sentiment, use_container_width=True)

with c2:
    st.subheader("Most Common Keywords")
    vectorizer = CountVectorizer(stop_words='english', max_features=15)
    all_text = " ".join(df["Message"])
    if all_text.strip():
        X = vectorizer.fit_transform([all_text])
        word_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out()).T
        word_freq.columns = ['frequency']
        word_freq = word_freq.sort_values(by='frequency', ascending=False)
        fig_keywords = px.bar(word_freq, x=word_freq.index, y='frequency', title="Top 15 Keywords")
        st.plotly_chart(fig_keywords, use_container_width=True)
    else:
        st.write("Not enough text to generate keywords.")

st.divider()

# --- Detailed Data View ---
st.header("Detailed Conversation Analysis")
with st.expander("Show Raw Data and Scores"):
    st.dataframe(df, use_container_width=True)
