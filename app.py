from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import requests
import re
import os
import pickle
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Startup cache ────────────────────────────────────────────
# Expensive computations (SVD, embeddings) are saved to disk on first run
# and reloaded instantly on subsequent starts.
CACHE_DIR = os.path.join(BASE_DIR, '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)

_SOURCE_FILES = [
    os.path.join(BASE_DIR, 'ml-25m/ratings_filtered.csv'),
    os.path.join(BASE_DIR, 'ml-25m/movies_filtered.csv'),
]

def _cache_fresh(path):
    """True if cache file exists and is newer than every source CSV."""
    if not os.path.exists(path):
        return False
    t = os.path.getmtime(path)
    return all(os.path.getmtime(s) <= t for s in _SOURCE_FILES)

SVD_CACHE = os.path.join(CACHE_DIR, 'svd.pkl')
EMB_CACHE = os.path.join(CACHE_DIR, 'embeddings.npy')
POP_CACHE = os.path.join(CACHE_DIR, 'popularity.npy')

TMDB_TOKEN = os.getenv('TMDB_TOKEN')
TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w500'

@lru_cache(maxsize=5000)
def fetch_tmdb_data(ml_title):
    """
    Given a MovieLens title like "Star Wars (1977)",
    search TMDB and return poster + description.
    Returns None fields if nothing is found.
    Cached in-process so the same title is never fetched twice per session.
    """
    # Parse "Star Wars (1977)" → title="Star Wars", year="1977"
    match = re.match(r'^(.*?)\s*\((\d{4})\)$', ml_title.strip())
    if match:
        title, year = match.group(1), match.group(2)
    else:
        title, year = ml_title, None

    params = {'query': title, 'language': 'en-US'}
    if year:
        params['year'] = year

    try:
        response = requests.get(
            'https://api.themoviedb.org/3/search/movie',
            headers={'Authorization': f'Bearer {TMDB_TOKEN}'},
            params=params,
            timeout=5  # don't hang forever if TMDB is slow
        )
        results = response.json().get('results', [])
        if not results:
            return {'poster': None, 'overview': None, 'tmdb_rating': None}

        movie = results[0]  # first result is usually the right one
        return {
            'poster': TMDB_IMAGE_BASE + movie['poster_path'] if movie.get('poster_path') else None,
            'overview': movie.get('overview') or None,
            'tmdb_rating': movie.get('vote_average') or None,
        }
    except requests.RequestException:
        return {'poster': None, 'overview': None, 'tmdb_rating': None}

def fetch_tmdb_batch(titles):
    """Fetch TMDB data for a list of titles in parallel."""
    with ThreadPoolExecutor(max_workers=8) as executor:
        return list(executor.map(fetch_tmdb_data, titles))

# ── Load data ────────────────────────────────────────────────
# ratings_filtered.csv is 614MB — only load it when caches need to be built.
movies = pd.read_csv(os.path.join(BASE_DIR, 'ml-25m/movies_filtered.csv'))

_need_ratings = not _cache_fresh(SVD_CACHE) or not _cache_fresh(POP_CACHE)
if _need_ratings:
    print("Loading ratings (needed to build cache)...")
    ratings = pd.read_csv(os.path.join(BASE_DIR, 'ml-25m/ratings_filtered.csv'))

# ── Content-based: genre similarity matrix ───────────────────
# ── Convert genre strings to binary columns ──────────────────
# "Adventure|Animation|Children" → adventure=1, animation=1, children=1, rest=0
#
# str.get_dummies('|') splits each string by '|' and creates one column
# per unique value, filling with 1 if present, 0 if not.
# So a movie with "Action|Comedy" gets action=1, comedy=1, drama=0, etc.
genre_dummies = movies['genres'].str.get_dummies('|')

# Drop movies with no genres listed
if '(no genres listed)' in genre_dummies.columns:
    genre_dummies = genre_dummies.drop(columns=['(no genres listed)'])

genre_columns = genre_dummies.columns.tolist()

# Attach the binary columns back to the movies dataframe
movies = pd.concat([movies, genre_dummies], axis=1)

genre_matrix = movies[genre_columns].values
similarity_matrix = cosine_similarity(genre_matrix)

# ── Popularity scores (used to rerank semantic search results) ─
if _cache_fresh(POP_CACHE):
    popularity_scores = np.load(POP_CACHE)
else:
    _rating_counts = ratings.groupby('movieId').size()
    movies['_pop'] = movies['movieId'].map(_rating_counts).fillna(0)
    _log_pop = np.log1p(movies['_pop'].values.astype(float))
    popularity_scores = (_log_pop - _log_pop.min()) / (_log_pop.max() - _log_pop.min() + 1e-9)
    np.save(POP_CACHE, popularity_scores)
    print("Popularity cache saved.")

# ── Collaborative: SVD matrix factorization ──────────────────
if _cache_fresh(SVD_CACHE):
    print("Loading SVD from cache...")
    with open(SVD_CACHE, 'rb') as f:
        _svd = pickle.load(f)
    predicted_df = _svd['predicted_df']
    user_movie_matrix = _svd['user_movie_matrix']
    print(f"SVD loaded ({len(predicted_df)} active users).")
else:
    ratings_per_user = ratings.groupby('userId').size()
    active_users = ratings_per_user[ratings_per_user >= 1000].index
    ratings_svd = ratings[ratings['userId'].isin(active_users)]
    print(f"Active users (1000+ ratings): {len(active_users)}")

    user_movie_matrix = ratings_svd.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    svd = TruncatedSVD(n_components=50, random_state=42)
    svd.fit(user_movie_matrix)

    user_factors  = svd.transform(user_movie_matrix)
    movie_factors = svd.components_
    predicted_ratings = np.dot(user_factors, movie_factors)

    predicted_df = pd.DataFrame(
        predicted_ratings,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.columns
    )
    print("SVD model trained. Saving to cache...")
    with open(SVD_CACHE, 'wb') as f:
        pickle.dump({'predicted_df': predicted_df, 'user_movie_matrix': user_movie_matrix}, f)
    print("Cache saved.")

# ── Semantic search: HuggingFace embeddings ──────────────────
print("Loading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def movie_to_text(row):
    active_genres = [g.replace('_', ' ') for g in genre_columns if row[g] == 1]
    genre_str = ', '.join(active_genres) if active_genres else 'unknown'
    return f"{row['title']} — {genre_str}"

movies['text'] = movies.apply(movie_to_text, axis=1)

if _cache_fresh(EMB_CACHE):
    print("Loading embeddings from cache...")
    movie_embeddings = np.load(EMB_CACHE)
    print(f"Embeddings loaded ({len(movie_embeddings)} movies).")
else:
    movie_embeddings = embed_model.encode(movies['text'].tolist(), show_progress_bar=True)
    np.save(EMB_CACHE, movie_embeddings)
    print("Embeddings saved to cache.")

# ── Recommender functions ────────────────────────────────────
def get_genres(row):
    return [g for g in genre_columns if row[g] == 1]

def recommend_similar_movies(movie_title, n_recommendations=5):
    try:
        movie_idx = movies[movies['title'] == movie_title].index[0]
    except IndexError:
        return None

    similarity_scores = similarity_matrix[movie_idx]
    similar_movie_indices = similarity_scores.argsort()[-n_recommendations-2:-2][::-1]

    rows = [movies.iloc[idx] for idx in similar_movie_indices]
    tmdb_data = fetch_tmdb_batch([row['title'] for row in rows])

    return [{'title': row['title'], 'similarity_score': float(similarity_scores[idx]),
             'genres': get_genres(row), **tmdb}
            for row, tmdb, idx in zip(rows, tmdb_data, similar_movie_indices)]

def recommend_movies_for_user(user_id, n_recommendations=5):
    if user_id not in predicted_df.index:
        return None

    user_predictions = predicted_df.loc[user_id]
    already_rated = user_movie_matrix.loc[user_id]
    user_predictions = user_predictions[already_rated == 0]
    top = user_predictions.nlargest(n_recommendations)

    rows, titles, scores = [], [], []
    for movie_id, score in top.items():
        row = movies[movies['movieId'] == movie_id]
        if row.empty:
            continue
        rows.append(row.iloc[0])
        titles.append(row['title'].values[0])
        scores.append(score)

    tmdb_data = fetch_tmdb_batch(titles)
    return [{'title': title, 'recommendation_score': round(float(score), 2),
             'genres': get_genres(row), **tmdb}
            for row, title, score, tmdb in zip(rows, titles, scores, tmdb_data)]

def semantic_search(query, n_recommendations=5):
    query_embedding = embed_model.encode([query])
    sem_scores = cosine_similarity(query_embedding, movie_embeddings)[0]

    # Blend semantic similarity (65%) with log-normalised popularity (35%).
    # Without this, vague genre queries ("comedy") can surface obscure films that
    # happen to be geometrically close in embedding space over well-known ones.
    blended = 0.65 * sem_scores + 0.35 * popularity_scores
    top_indices = blended.argsort()[-n_recommendations:][::-1]

    rows = [movies.iloc[idx] for idx in top_indices]
    tmdb_data = fetch_tmdb_batch([row['title'] for row in rows])

    return [{'title': row['title'], 'similarity_score': float(sem_scores[idx]),
             'genres': get_genres(row), **tmdb}
            for row, tmdb, idx in zip(rows, tmdb_data, top_indices)]

# Create Flask app
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)

# API endpoints
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'movies': len(movies), 'users': len(ratings['userId'].unique())})

@app.route('/api/movies/autocomplete', methods=['GET'])
def autocomplete():
    q = request.args.get('q', '').strip().lower()
    if len(q) < 2:
        return jsonify({'titles': []})
    matches = movies[movies['title'].str.lower().str.contains(q, regex=False)]['title'].head(12).tolist()
    return jsonify({'titles': matches})

@app.route('/api/movies/<movie_title>', methods=['GET'])
def get_similar(movie_title):
    n = request.args.get('n', 5, type=int)
    recs = recommend_similar_movies(movie_title, n_recommendations=n)
    if recs is None:
        return jsonify({'error': f'Movie not found: {movie_title}'}), 404
    return jsonify({'movie': movie_title, 'recommendations': recs})

@app.route('/api/recommend/user/<int:user_id>', methods=['GET'])
def recommend_user(user_id):
    n = request.args.get('n', 5, type=int)
    recs = recommend_movies_for_user(user_id, n_recommendations=n)
    if recs is None:
        return jsonify({'error': f'User not found: {user_id}'}), 404
    return jsonify({'user_id': user_id, 'recommendations': recs})

@app.route('/api/search', methods=['GET'])
def search():
    # e.g. GET /api/search?q=funny+movie+for+kids&n=5
    query = request.args.get('q', '').strip()
    n = request.args.get('n', 5, type=int)
    if not query:
        return jsonify({'error': 'Missing query parameter q'}), 400
    recs = semantic_search(query, n_recommendations=n)
    return jsonify({'query': query, 'recommendations': recs})

@app.route('/api/users/sample', methods=['GET'])
def sample_users():
    sample_ids = [int(uid) for uid in predicted_df.index[:8].tolist()]
    return jsonify({'user_ids': sample_ids})

if __name__ == '__main__':
    print("🚀 Starting API at http://localhost:8000")
    app.run(debug=True, port=8000)