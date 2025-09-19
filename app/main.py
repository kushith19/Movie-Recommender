import streamlit as st
import pandas as pd
from preprocessing import load_data
from recommender import create_similarity_matrix, recommend_movie, save_similarity_matrix, load_similarity_matrix
import os

# -------------------------------
# Project Paths
# -------------------------------
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')

# -------------------------------
# Load Movies Data
# -------------------------------
@st.cache_data
def get_movies():
    """
    Load movies.csv safely.
    Falls back to sample_movies.csv in the same folder as preprocessing.py if full dataset is not available.
    """
    try:
        df = load_data()  # load_data now handles both full or sample datasets
        # st.write(f"✅ Loaded dataset with {len(df)} movies")
        return df
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
        return pd.DataFrame()

# -------------------------------
# Load or Create Similarity Matrix
# -------------------------------
@st.cache_resource
def get_similarity(df):
    """
    Load or create similarity matrix.
    Saves it in models/cosine_sim.pkl
    """
    sim_path = os.path.join(MODELS_DIR, 'cosine_sim.pkl')
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.exists(sim_path):
        # st.write("⚡ Loading existing similarity matrix...")
        return load_similarity_matrix(sim_path)
    else:
        # st.write("⚡ Creating similarity matrix (this may take a few seconds)...")
        sim = create_similarity_matrix(df)
        save_similarity_matrix(sim, sim_path)
        return sim

# -------------------------------
# Load Data & Similarity Matrix
# -------------------------------
df = get_movies()
cosine_sim = get_similarity(df)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎬 Movie Recommender System")
st.sidebar.header("Filter Movies")

# Genres filter
genres_list = sorted({g for sublist in df['genres'].dropna().str.split() for g in sublist if g})
selected_genres = st.sidebar.multiselect("Select Genres", genres_list)

# Year filter
valid_years = df[df['release_year'] > 1900]['release_year']
min_year, max_year = int(valid_years.min()), int(valid_years.max())
year_range = st.sidebar.slider("Select Release Year Range", min_year, max_year, (min_year, max_year))

# Popularity filter
min_pop, max_pop = float(df['popularity'].min()), float(df['popularity'].max())
pop_range = st.sidebar.slider("Select Popularity Range", min_pop, max_pop, (min_pop, max_pop))

# Movie selection
movie_list = df['title'].dropna().unique()
selected_movie = st.selectbox("Choose a movie you like:", movie_list)

# -------------------------------
# Recommendation Button
# -------------------------------
if st.button("Recommend"):
    # Get top 50 recommendations
    recommendations = recommend_movie(selected_movie, df, cosine_sim, top_n=50)

    # Apply filters
    filtered_movies = df[df['title'].isin(recommendations)].copy()
    if selected_genres:
        filtered_movies = filtered_movies[
            filtered_movies['genres'].apply(lambda g: any(genre in g.split() for genre in selected_genres))
        ]
    filtered_movies = filtered_movies[
        (filtered_movies['release_year'] >= year_range[0]) &
        (filtered_movies['release_year'] <= year_range[1]) &
        (filtered_movies['popularity'] >= pop_range[0]) &
        (filtered_movies['popularity'] <= pop_range[1])
    ]

    # Display filtered recommendations
    st.subheader("Movies you may like:")
    if not filtered_movies.empty:
        for idx, row in enumerate(filtered_movies.head(10).itertuples(), start=1):
            st.write(f"{idx}. {row.title} ({row.release_year})")
    else:
        st.write("No movies match the selected filters.")


# st.sidebar.write("**Debug Info**")
# st.sidebar.write(f"Dataset rows: {len(df)}")
# st.sidebar.write(f"First movie: {df['title'].iloc[0] if len(df) > 0 else 'N/A'}")
# st.sidebar.write(f"Similarity matrix shape: {cosine_sim.shape if cosine_sim is not None else 'N/A'}")
