import pandas as pd
import ast
import os

def preprocess_and_save(full=True):
    """
    Preprocess TMDB datasets and save as movies.csv.
    If full=False or datasets not found, use a small sample dataset for deployment/demo.
    """
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    movies_path = os.path.join(BASE_DIR, 'data', 'tmdb_5000_movies.csv')
    credits_path = os.path.join(BASE_DIR, 'data', 'tmdb_5000_credits.csv')
    output_path = os.path.join(BASE_DIR, 'data', 'movies.csv')

    if full and os.path.exists(movies_path) and os.path.exists(credits_path):
        movies = pd.read_csv(movies_path)
        credits = pd.read_csv(credits_path)
    else:
        # Use a small sample dataset for deployment/demo
        sample_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'app_data', 'sample_movies.csv'))
        if not os.path.exists(sample_path):
            raise FileNotFoundError(
                "Sample dataset not found. Please create 'sample_movies.csv' in app_data/ for deployment."
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

    df_final.to_csv(output_path, index=False)
    print(f"✅ Preprocessed dataset saved to {output_path}")
    return df_final


def load_data():
    """
    Load movies.csv if exists. Otherwise, fallback to sample dataset for deployment.
    """
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path = os.path.join(BASE_DIR, 'data', 'movies.csv')

    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        # Use sample dataset from app_data for deployment/demo
        sample_path = os.path.abspath(os.path.join(BASE_DIR, '..', 'app_data', 'sample_movies.csv'))
        if not os.path.exists(sample_path):
            raise FileNotFoundError(
                "Sample dataset not found. Please create 'sample_movies.csv' in app_data/ for deployment."
            )
        print("⚡ Using sample dataset for deployment/demo.")
        return pd.read_csv(sample_path)


if __name__ == "__main__":
    preprocess_and_save()
