from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import requests
import re
import os
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Startup cache ────────────────────────────────────────────
CACHE_DIR = os.path.join(BASE_DIR, '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)

_SOURCE_FILES = [
    os.path.join(BASE_DIR, 'ml-25m/ratings_filtered.csv'),
    os.path.join(BASE_DIR, 'ml-25m/movies_filtered.csv'),
]

def _cache_fresh(path):
    if not os.path.exists(path):
        return False
    t = os.path.getmtime(path)
    existing = [s for s in _SOURCE_FILES if os.path.exists(s)]
    if not existing:
        return True  # no source files on this machine, trust the cache
    return all(os.path.getmtime(s) <= t for s in existing)

POP_CACHE = os.path.join(CACHE_DIR, 'popularity.npy')

TMDB_TOKEN = os.getenv('TMDB_TOKEN')
TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w500'

@lru_cache(maxsize=5000)
def fetch_tmdb_data(ml_title):
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
            timeout=5
        )
        results = response.json().get('results', [])
        if not results:
            return {'poster': None, 'overview': None, 'tmdb_rating': None}
        movie = results[0]
        return {
            'poster': TMDB_IMAGE_BASE + movie['poster_path'] if movie.get('poster_path') else None,
            'overview': movie.get('overview') or None,
            'tmdb_rating': movie.get('vote_average') or None,
        }
    except requests.RequestException:
        return {'poster': None, 'overview': None, 'tmdb_rating': None}

def fetch_tmdb_batch(titles):
    with ThreadPoolExecutor(max_workers=8) as executor:
        return list(executor.map(fetch_tmdb_data, titles))

# ── Load data ────────────────────────────────────────────────
movies = pd.read_csv(os.path.join(BASE_DIR, 'ml-25m/movies_filtered.csv'))

# ── Genre similarity matrix ──────────────────────────────────
genre_dummies = movies['genres'].str.get_dummies('|')
if '(no genres listed)' in genre_dummies.columns:
    genre_dummies = genre_dummies.drop(columns=['(no genres listed)'])
genre_columns = genre_dummies.columns.tolist()
movies = pd.concat([movies, genre_dummies], axis=1)
genre_matrix = movies[genre_columns].values.astype('float32')

# ── Popularity scores ────────────────────────────────────────
RATINGS_FILE = os.path.join(BASE_DIR, 'ml-25m/ratings_filtered.csv')

if _cache_fresh(POP_CACHE):
    popularity_scores = np.load(POP_CACHE)
elif os.path.exists(RATINGS_FILE):
    print("Loading ratings to build popularity cache...")
    ratings = pd.read_csv(RATINGS_FILE)
    _rating_counts = ratings.groupby('movieId').size()
    movies['_pop'] = movies['movieId'].map(_rating_counts).fillna(0)
    _log_pop = np.log1p(movies['_pop'].values.astype(float))
    popularity_scores = (_log_pop - _log_pop.min()) / (_log_pop.max() - _log_pop.min() + 1e-9)
    np.save(POP_CACHE, popularity_scores)
    print("Popularity cache saved.")
else:
    print("No ratings file — using uniform popularity.")
    popularity_scores = np.ones(len(movies))

# ── TF-IDF search index ──────────────────────────────────────
print("Building TF-IDF search index...")

def movie_to_text(row):
    active_genres = [g.replace('_', ' ') for g in genre_columns if row[g] == 1]
    genre_str = ' '.join(active_genres) if active_genres else 'unknown'
    title_clean = re.sub(r'\(\d{4}\)', '', row['title']).strip()
    # Repeat genres to boost their weight relative to title words
    return f"{title_clean} {genre_str} {genre_str}"

movies['text'] = movies.apply(movie_to_text, axis=1)
tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
tfidf_matrix = tfidf.fit_transform(movies['text'])
print(f"TF-IDF index built ({tfidf_matrix.shape[0]} movies, {tfidf_matrix.shape[1]} terms).")

# ── Query synonym expansion ──────────────────────────────────
_SYNONYMS = {
    'funny': 'comedy', 'hilarious': 'comedy', 'humorous': 'comedy', 'humor': 'comedy',
    'laugh': 'comedy', 'lighthearted': 'comedy', 'feel good': 'comedy drama',
    'feel-good': 'comedy drama', 'cozy': 'comedy drama romance', 'rainy': 'comedy drama',
    'scary': 'horror', 'frightening': 'horror', 'spooky': 'horror', 'creepy': 'horror',
    'romantic': 'romance', 'love story': 'romance', 'date night': 'romance comedy',
    'thrilling': 'thriller', 'suspenseful': 'thriller', 'suspense': 'thriller',
    'tense': 'thriller', 'intense': 'thriller drama',
    'action-packed': 'action', 'explosive': 'action',
    'animated': 'animation', 'cartoon': 'animation',
    'kids': 'animation children', 'children': 'children animation',
    'family': 'children comedy', 'family friendly': 'children comedy animation',
    'science fiction': 'sci-fi', 'scifi': 'sci-fi', 'futuristic': 'sci-fi',
    'space': 'sci-fi adventure', 'dystopian': 'sci-fi drama',
    'documentary': 'documentary', 'doc': 'documentary', 'true story': 'documentary drama',
    'historical': 'history drama', 'period': 'drama history',
    'mystery': 'mystery thriller', 'whodunit': 'mystery crime',
    'dark': 'crime thriller drama', 'gritty': 'crime drama',
    'war': 'war drama', 'combat': 'war action',
    'adventure': 'adventure action', 'epic': 'adventure drama',
    'musical': 'musical', 'music': 'musical',
    'western': 'western', 'cowboy': 'western',
}

def expand_query(query):
    q = query.lower()
    extra = []
    for phrase, genres in _SYNONYMS.items():
        if phrase in q:
            extra.append(genres)
    if extra:
        return query + ' ' + ' '.join(extra)
    return query

# ── Genome + CF factors ──────────────────────────────────────
_GENOME_FACTORS = os.path.join(CACHE_DIR, 'genome_factors.npy')
_GENOME_MASK    = os.path.join(CACHE_DIR, 'genome_mask.npy')
_CF_FACTORS     = os.path.join(CACHE_DIR, 'cf_item_factors.npy')

if os.path.exists(_GENOME_FACTORS) and os.path.exists(_GENOME_MASK):
    genome_factors = np.load(_GENOME_FACTORS)
    genome_mask    = np.load(_GENOME_MASK)
    print(f"Genome factors loaded ({genome_mask.sum()} movies).")
else:
    genome_factors = None
    genome_mask    = None
    print("No genome factors — using genre-only similarity.")

if os.path.exists(_CF_FACTORS):
    cf_item_factors = np.load(_CF_FACTORS)
    print(f"CF item factors loaded {cf_item_factors.shape}.")
else:
    cf_item_factors = None
    print("No CF factors — collaborative filtering disabled.")

# ── Recommender functions ────────────────────────────────────
def get_genres(row):
    return [g for g in genre_columns if row[g] == 1]

def recommend_similar_movies(movie_title, n_recommendations=5):
    try:
        movie_idx = movies[movies['title'] == movie_title].index[0]
    except IndexError:
        return None

    # Genre similarity (always available)
    genre_sim = cosine_similarity(genre_matrix[movie_idx].reshape(1, -1), genre_matrix)[0]

    # Genome similarity
    has_genome = genome_factors is not None and genome_mask[movie_idx]
    if has_genome:
        genome_sim = cosine_similarity(genome_factors[movie_idx].reshape(1, -1), genome_factors)[0]
        genome_sim[~genome_mask] = 0
    else:
        genome_sim = None

    # Collaborative filtering similarity
    if cf_item_factors is not None:
        cf_sim = cosine_similarity(cf_item_factors[movie_idx].reshape(1, -1), cf_item_factors)[0]
    else:
        cf_sim = None

    # Hybrid blend
    if genome_sim is not None and cf_sim is not None:
        scores = 0.15 * genre_sim + 0.50 * genome_sim + 0.35 * cf_sim
    elif genome_sim is not None:
        scores = 0.25 * genre_sim + 0.75 * genome_sim
    elif cf_sim is not None:
        scores = 0.30 * genre_sim + 0.70 * cf_sim
    else:
        scores = genre_sim

    scores[movie_idx] = -1  # exclude self
    top_indices = scores.argsort()[-n_recommendations:][::-1]

    rows = [movies.iloc[idx] for idx in top_indices]
    tmdb_data = fetch_tmdb_batch([row['title'] for row in rows])

    return [{'title': row['title'], 'similarity_score': float(scores[idx]),
             'genres': get_genres(row), **tmdb}
            for row, tmdb, idx in zip(rows, tmdb_data, top_indices)]

def semantic_search(query, n_recommendations=5):
    expanded = expand_query(query)
    query_vec = tfidf.transform([expanded])
    sem_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    blended = 0.65 * sem_scores + 0.35 * popularity_scores
    top_indices = blended.argsort()[-n_recommendations:][::-1]

    rows = [movies.iloc[idx] for idx in top_indices]
    tmdb_data = fetch_tmdb_batch([row['title'] for row in rows])

    return [{'title': row['title'], 'similarity_score': float(blended[idx]),
             'genres': get_genres(row), **tmdb}
            for row, tmdb, idx in zip(rows, tmdb_data, top_indices)]

# ── Flask app ────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'movies': len(movies)})

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

@app.route('/api/search', methods=['GET'])
def search():
    query = request.args.get('q', '').strip()
    n = request.args.get('n', 5, type=int)
    if not query:
        return jsonify({'error': 'Missing query parameter q'}), 400
    recs = semantic_search(query, n_recommendations=n)
    return jsonify({'query': query, 'recommendations': recs})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
