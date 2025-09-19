import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_similarity_matrix(df):
    """Create a cosine similarity matrix using TF-IDF on combined features."""
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined_features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def recommend_movie(movie_title, df, cosine_sim, top_n=10):
    """Return top_n recommended movie titles for a given movie title."""
    if movie_title not in df["title"].values:
        return []  # movie not found
    
    idx = df.index[df["title"] == movie_title][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # skip the first one (it's the movie itself)
    sim_scores = sim_scores[1 : top_n + 1]
    movie_indices = [i[0] for i in sim_scores]

    return df["title"].iloc[movie_indices].values.tolist()

def save_similarity_matrix(cosine_sim, path="models/cosine_sim.pkl"):
    """Save the cosine similarity matrix to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cosine_sim, f)

def load_similarity_matrix(path="models/cosine_sim.pkl"):
    """Load a cosine similarity matrix from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)
