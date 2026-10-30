from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from review_insights.baseline import train
from review_insights.data_gen import make_dataset

st.set_page_config(page_title="Insights de avaliações", page_icon="📊", layout="wide")
st.title("📊 Insights de avaliações de clientes")

DATA = Path(__file__).parent / "data" / "reviews.csv"
upload = st.sidebar.file_uploader("CSV com colunas: review, date (opcional: sentiment, topic)", type="csv")
df = pd.read_csv(upload) if upload else (pd.read_csv(DATA) if DATA.exists() else make_dataset())
if upload:
    st.sidebar.caption("Para rotular, o app treina o baseline com o dataset de exemplo.")

@st.cache_resource
def baseline():
    ref = pd.read_csv(DATA) if DATA.exists() else make_dataset()
    return train(ref["review"], ref["sentiment"])

use_llm = st.sidebar.checkbox("Classificar com LLM (usa a API)", value=False)
if "sentiment" not in df or upload:
    df["sentiment"] = baseline().predict(df["review"])
if use_llm:
    from review_insights.llm_classifier import classify

    with st.spinner("Classificando com LLM..."):
        res = classify(df["review"].tolist())
    df["sentiment"], df["topic"] = res.sentiments, res.topics
if "topic" not in df:
    df["topic"] = "n/d"
if "date" in df:
    df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m")

c1, c2, c3 = st.columns(3)
c1.metric("Avaliações", len(df))
c2.metric("% positivas", f"{(df.sentiment == 'positive').mean():.0%}")
c3.metric("% negativas", f"{(df.sentiment == 'negative').mean():.0%}")

colors = alt.Scale(domain=["negative", "neutral", "positive"], range=["#d62728", "#7f7f7f", "#2ca02c"])
left, right = st.columns(2)
with left:
    st.subheader("Sentimento por tópico")
    st.altair_chart(
        alt.Chart(df).mark_bar().encode(
            x=alt.X("topic:N", title="Tópico"), y=alt.Y("count():Q", title="Avaliações"),
            color=alt.Color("sentiment:N", scale=colors), tooltip=["topic", "sentiment", "count()"],
        ), width="stretch")
with right:
    if "month" in df:
        st.subheader("Evolução mensal")
        monthly = df.groupby(["month", "sentiment"]).size().reset_index(name="n")
        st.altair_chart(
            alt.Chart(monthly).mark_line(point=True).encode(
                x="month:N", y="n:Q", color=alt.Color("sentiment:N", scale=colors)), width="stretch")

st.subheader("Principais reclamações")
negative = df[df.sentiment == "negative"]
st.dataframe(negative.groupby("topic").size().sort_values(ascending=False).rename("reclamações"))
st.dataframe(negative[["review", "topic"]].head(20), width="stretch")
