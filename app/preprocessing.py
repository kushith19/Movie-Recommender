import pandas as pd
import ast
import os

# -------------------------------
# Directories
# -------------------------------
# Project root (one level up from this file)
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
APP_DIR = os.path.abspath(os.path.dirname(__file__))  # fallback to same folder for sample dataset

# -------------------------------
# Preprocessing
# -------------------------------
def preprocess_and_save(full=True):
    """
    Preprocess TMDB datasets and save as movies.csv.
    If full=False or datasets not found, use sample_movies.csv from the same folder for deployment/demo.
    """
    movies_path = os.path.join(DATA_DIR, 'tmdb_5000_movies.csv')
    credits_path = os.path.join(DATA_DIR, 'tmdb_5000_credits.csv')
    output_path = os.path.join(DATA_DIR, 'movies.csv')

    if full and os.path.exists(movies_path) and os.path.exists(credits_path):
        print(f"⚡ Using full dataset from {DATA_DIR}")
        movies = pd.read_csv(movies_path)
        credits = pd.read_csv(credits_path)
    else:
        # Use sample dataset for deployment/demo
        sample_path = os.path.join(APP_DIR, 'sample_movies.csv')
        print(f"DEBUG: Looking for sample dataset at {sample_path}")
        if not os.path.exists(sample_path):
            raise FileNotFoundError(
                "Sample dataset not found. Please create 'sample_movies.csv' in the app folder for deployment."
            )
        print("⚡ Using sample dataset for deployment/demo.")
        return pd.read_csv(sample_path)

    # Clean column names
    movies.rename(columns=lambda x: x.strip(), inplace=True)
    credits.rename(columns=lambda x: x.strip(), inplace=True)

    # Ensure 'title' exists
    if 'title' not in movies.columns and 'original_title' in movies.columns:
        movies.rename(columns={'original_title': 'title'}, inplace=True)

    # Merge datasets
    credits.rename(columns={'movie_id': 'id'}, inplace=True)
    df = movies.merge(credits, on='id', how='inner')

    # Helper function to extract names from JSON-like columns
    def extract_names(obj):
        try:
            lst = ast.literal_eval(obj)
            return ' '.join([i.get('name', '') for i in lst])
        except:
            return ''

    df['genres'] = df['genres'].apply(extract_names)
    df['keywords'] = df['keywords'].apply(extract_names)
    df['cast'] = df['cast'].apply(lambda x: ' '.join([i.get('name', '') for i in ast.literal_eval(x)][:5]))
    df['crew'] = df['crew'].apply(lambda x: ' '.join([i.get('name', '') for i in ast.literal_eval(x) if i.get('job') == 'Director']))

    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0)
    df['combined_features'] = (
        df['overview'].fillna('') + ' ' + df['genres'] + ' ' + df['keywords'] + ' ' + df['cast'] + ' ' + df['crew']
    )

    # Poster URLs
    df['poster_path'] = df.get('poster_path', '').fillna('')
    df['poster_url'] = df['poster_path'].apply(
        lambda x: f"https://image.tmdb.org/t/p/w500{x}" if x else "https://via.placeholder.com/150x225?text=No+Image"
    )

    # Keep only useful columns
    df_final = df[['id', 'title', 'genres', 'release_year', 'popularity', 'poster_url', 'combined_features']]

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"✅ Preprocessed dataset saved to {output_path}")
    return df_final

# -------------------------------
# Load Data
# -------------------------------
def load_data():
    """
    Load movies.csv if exists. Otherwise, fallback to sample dataset from the app folder.
    """
    movies_csv_path = os.path.join(DATA_DIR, 'movies.csv')
    sample_path = os.path.join(APP_DIR, 'sample_movies.csv')

    print(f"DEBUG: movies_csv_path = {movies_csv_path}")
    print(f"DEBUG: sample_path = {sample_path}")
    print(f"DEBUG: movies.csv exists? {os.path.exists(movies_csv_path)}")
    print(f"DEBUG: sample_movies.csv exists? {os.path.exists(sample_path)}")

    if os.path.exists(movies_csv_path):
        return pd.read_csv(movies_csv_path)
    elif os.path.exists(sample_path):
        print("⚡ Using sample dataset for deployment/demo.")
        return pd.read_csv(sample_path)
    else:
        raise FileNotFoundError(
            "Sample dataset not found. Please create 'sample_movies.csv' in the app folder for deployment."
        )

# -------------------------------
# Run Preprocessing Standalone
# -------------------------------
if __name__ == "__main__":
    preprocess_and_save()
