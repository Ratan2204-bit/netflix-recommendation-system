# 🎬 Netflix Recommendation System 

An advanced and optimizeed Netflix recommendation system built using content-based filtering , NLP techniques , and hybrid scoring . The system recommends relevant movies and TV shows using metadata such as genre , cast , director , and description .

---

## 🚀 Features 
- Content based recommendation using TF-IDF & cosine similarity
- Weighted feature engineerng for better relevance
- Popularity-aware hybrid recommendation
- Unsupervised clustering (K-Means)
- Explainable recommendations
- Optimized for low-memory systems
- Interactive Streamlit web applications 

---

## 🧠 Tech Stack
- Python
- Pandas , Numpy
- Scikit-learn
- NLP (TF-IDF , n-grams)
- Streamlit

---

## 📂 Project Structure
Netflix-Recommendation-System/
├── data/
│ ├── raw/
│ │ └── netflix_titles.csv # Original dataset
│ └── processed/
│ └── netflix_cleaned.csv # Cleaned & feature-engineered data
│
├── notebooks/
│ ├── 01_data_exploration.ipynb # EDA and initial analysis
│ ├── 02_feature_engineering.ipynb # Feature creation and experiments
│ ├── 03_model_building.ipynb # TF-IDF, similarity, clustering
│ └── 04_evaluation.ipynb # Model evaluation & analysis
│
├── src/
│ ├── init.py
│ ├── data_preprocessing.py # Data cleaning 
preprocessing
│ ├── feature_engineering.py # Weighted metadata feature creation
│ ├── vectorization.py # TF-IDF vectorization logic
│ ├── clustering.py # K-Means clustering
│ ├── recommender.py # Recommendation & hybrid scoring logic
│ └── utils.py # Helper functions
│
├── models/
│ ├── tfidf_vectorizer.pkl # Saved TF-IDF model
│ ├── cosine_similarity.npy # Precomputed similarity matrix
│ └── kmeans_model.pkl # Trained clustering model
│
├── app.py # Streamlit web application
├── requirements.txt # Project dependencies
├── README.md # Project documentation
├── .gitignore # Git ignore rules
└── setup.py # Package configuration