from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer
import requests
import re
import os
from dotenv import load_dotenv

load_dotenv()
os.chdir('/Users/hilimor/movie-recommender')

TMDB_TOKEN = os.getenv('TMDB_TOKEN')
TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w500'

def fetch_tmdb_data(ml_title):
    """
    Given a MovieLens title like "Star Wars (1977)",
    search TMDB and return poster + description.
    Returns None fields if nothing is found.
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

# ── Load data ────────────────────────────────────────────────
ratings = pd.read_csv('ml-25m/ratings_filtered.csv')

movies = pd.read_csv('ml-25m/movies_filtered.csv')

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

# ── Collaborative: SVD matrix factorization ──────────────────
# Step 1: filter to active users only (rated 50+ movies)
# The full dataset has 162k users × 13k movies = too large to fit in memory
# Active users have richer taste profiles anyway — better recommendations
ratings_per_user = ratings.groupby('userId').size()
active_users = ratings_per_user[ratings_per_user >= 1000].index
ratings_svd = ratings[ratings['userId'].isin(active_users)]

print(f"Active users (1000+ ratings): {len(active_users)}")

# Step 2: build a users × movies matrix of ratings (0 = not rated)
user_movie_matrix = ratings_svd.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Step 2: decompose into latent factors
# n_components = how many hidden "taste dimensions" to find
# Think of these as unlabeled concepts like "loves action", "prefers drama", etc.
svd = TruncatedSVD(n_components=50, random_state=42)
svd.fit(user_movie_matrix)

# Step 3: reconstruct the full matrix — this fills in predicted ratings
# for every (user, movie) pair, including ones we've never seen
user_factors  = svd.transform(user_movie_matrix)   # shape: (users, 50)
movie_factors = svd.components_                     # shape: (50, movies)
predicted_ratings = np.dot(user_factors, movie_factors)  # shape: (users, movies)

# Wrap in a DataFrame so we can look up by userId / movieId
predicted_df = pd.DataFrame(
    predicted_ratings,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.columns
)

print("SVD model trained.")

# ── Semantic search: HuggingFace embeddings ──────────────────
# Load a small but powerful sentence embedding model (~90MB, cached after first run)
# "all-MiniLM-L6-v2" maps any text to a 384-dimensional vector
print("Loading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Build a text description for each movie from its title + active genres.
# This is what the model will embed — richer text = better semantic matching.
def movie_to_text(row):
    active_genres = [g.replace('_', ' ') for g in genre_columns if row[g] == 1]
    genre_str = ', '.join(active_genres) if active_genres else 'unknown'
    return f"{row['title']} — {genre_str}"

movies['text'] = movies.apply(movie_to_text, axis=1)

# Embed all 1682 movies once at startup — shape: (1682, 384)
# This takes ~5s on CPU and is reused for every search query
movie_embeddings = embed_model.encode(movies['text'].tolist(), show_progress_bar=False)
print("Embedding model ready.")

# ── Recommender functions ────────────────────────────────────
def recommend_similar_movies(movie_title, n_recommendations=5):
    try:
        movie_idx = movies[movies['title'] == movie_title].index[0]
    except IndexError:
        return None

    similarity_scores = similarity_matrix[movie_idx]
    similar_movie_indices = similarity_scores.argsort()[-n_recommendations-2:-2][::-1]

    return [{'title': movies.iloc[idx]['title'], 'similarity_score': float(similarity_scores[idx]),
             **fetch_tmdb_data(movies.iloc[idx]['title'])}
            for idx in similar_movie_indices]

def recommend_movies_for_user(user_id, n_recommendations=5):
    if user_id not in predicted_df.index:
        return None

    # Get this user's predicted ratings for every movie
    user_predictions = predicted_df.loc[user_id]

    # Zero out movies they've already rated — don't recommend those
    already_rated = user_movie_matrix.loc[user_id]
    user_predictions = user_predictions[already_rated == 0]

    # Take top N by predicted rating
    top = user_predictions.nlargest(n_recommendations)

    results = []
    for movie_id, score in top.items():
        row = movies[movies['movieId'] == movie_id]
        if row.empty:
            continue
        title = row['title'].values[0]
        results.append({
            'title': title,
            'recommendation_score': round(float(score), 2),
            **fetch_tmdb_data(title)
        })
    return results

def semantic_search(query, n_recommendations=5):
    # Embed the user's query using the same model — same vector space as movies
    query_embedding = embed_model.encode([query])

    # Compute cosine similarity between the query and every movie embedding
    scores = cosine_similarity(query_embedding, movie_embeddings)[0]

    # Get top N indices sorted by score
    top_indices = scores.argsort()[-n_recommendations:][::-1]

    return [
        {
            'title': movies.iloc[idx]['title'],
            'similarity_score': float(scores[idx]),
            **fetch_tmdb_data(movies.iloc[idx]['title'])
        }
        for idx in top_indices
    ]

# Create Flask app
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to Movie Recommender API!', 'status': 'running'})

# API endpoints
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'movies': len(movies), 'users': len(ratings['userId'].unique())})

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

if __name__ == '__main__':
    print("🚀 Starting API at http://localhost:8000")
    app.run(debug=True, port=8000)