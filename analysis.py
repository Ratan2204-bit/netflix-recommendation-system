import pandas as pd
import numpy as np

df = pd.read_csv("data/netflix_titles.csv")

features = ["listed_in" , "description" , "cast" , "director" , "country" , "rating" , "type"]
df[features] = df[features].fillna('')

df["combined_features"] = ((df["listed_in"] + ' ')* 4 + (df["cast"] + ' ')* 2 + (df["director"]+ ' ')* 2 + df["description"]).str.lower()

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    stop_words = "english",
    ngram_range=(1,2),
    max_df=0.80,
    min_df=5,
    max_features=5000
)

tfidf_matrix = tfidf.fit_transform(df["combined_features"])

from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_matrix , tfidf_matrix)
indices = pd.Series(df.index , index=df["title"]).drop_duplicates()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=10 , random_state=42)
df["cluster"] = kmeans.fit_predict(tfidf_matrix)

df["popularity_score"] = df["release_year"].rank(pct=True)

# Final Score = 0.7 * Similarity + 0.3 * Popularity

def recommend(title , n=5):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores , key=lambda x: x[1] , reverse=True)[1:50]

    scores = []
    for i , sim in sim_scores:
        final_score = (0.85 * sim) + (0.15*df.iloc[i]["popularity_score"])
        scores.append((i , final_score))

    scores = sorted(scores , key=lambda x:x[1] , reverse=True)[:n]
    return df["title"].iloc[[i[0] for i in scores]]

def explain(title1 , title2):
    t1 = set(df[df["title"]==title1]["listed_in"].values[0].split(','))
    t2 = set(df[df["title"]==title2]["listed_in"].values[0].split(','))
    return t1.intersection(t2)

def debug_recommend(title):
    idx = indices[title]
    sim_scores = sorted(
        enumerate(cosine_sim[idx]),
        key = lambda x: x[1],
        reverse=True
    )[1:6]

    for i , score in sim_scores:
        print(df.iloc[i]["title"], "->" , round(score , 3))


import streamlit as st

st.title("🎬Netflix Recommendation System")

movie = st.selectbox("Select a title", df["title"].values)

if st.button("Recommend"):
    for rec in recommend(movie):
        st.write("👉", rec)