import streamlit as st
import pandas as pd
from preprocessing import load_data
from recommender import create_similarity_matrix, recommend_movie, save_similarity_matrix, load_similarity_matrix
import os

@st.cache_data
def get_movies():
    """
    Load movies.csv safely.
    Falls back to sample_movies.csv in app_data/ if full dataset is not available.
    """
    try:
        return load_data()  # load_data now handles full or sample dataset
    except FileNotFoundError as e:
        st.error(str(e))
        return pd.DataFrame()

@st.cache_resource
def get_similarity(df):
    """
    Load or create similarity matrix.
    Saves it in models/cosine_sim.pkl
    """
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sim_path = os.path.join(BASE_DIR, 'models', 'cosine_sim.pkl')

    if os.path.exists(sim_path):
        return load_similarity_matrix(sim_path)
    else:
        sim = create_similarity_matrix(df)
        os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)
        save_similarity_matrix(sim, sim_path)
        return sim

# Load movies dataframe
df = get_movies()
if df.empty:
    st.stop()

# Load or create similarity matrix
cosine_sim = get_similarity(df)

st.title("🎬 Movie Recommender System")
st.sidebar.header("Filter Movies")

# Sidebar filters
genres_list = sorted({g for sublist in df['genres'].dropna().str.split() for g in sublist if g})
selected_genres = st.sidebar.multiselect("Select Genres", genres_list)

valid_years = df[df['release_year'] > 1900]['release_year']
min_year, max_year = int(valid_years.min()), int(valid_years.max())
year_range = st.sidebar.slider("Select Release Year Range", min_year, max_year, (min_year, max_year))

min_pop, max_pop = float(df['popularity'].min()), float(df['popularity'].max())
pop_range = st.sidebar.slider("Select Popularity Range", min_pop, max_pop, (min_pop, max_pop))

movie_list = df['title'].dropna().unique()
selected_movie = st.selectbox("Choose a movie you like:", movie_list)

# Recommendation button
if st.button("Recommend"):
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

    # Display recommendations
    st.subheader("Movies you may like:")
    if not filtered_movies.empty:
        for idx, row in enumerate(filtered_movies.head(10).itertuples(), start=1):
            st.write(f"{idx}. {row.title} ({row.release_year})")
    else:
        st.write("No movies match the selected filters.")
