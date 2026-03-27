from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
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
ratings = pd.read_csv('ml-100k/u.data', sep='\t', header=None,
                      names=['userId', 'movieId', 'rating', 'timestamp'])

movies = pd.read_csv('ml-100k/u.item', sep='|', header=None, encoding='latin-1',
                     names=['movieId', 'title', 'releaseDate', 'videoReleaseDate', 'url',
                            'unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy',
                            'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror',
                            'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western'])

# ── Content-based: genre similarity matrix ───────────────────
genre_columns = ['unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy',
                 'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror',
                 'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western']

genre_matrix = movies[genre_columns].values
similarity_matrix = cosine_similarity(genre_matrix)

# ── Collaborative: SVD matrix factorization ──────────────────
# Step 1: build a users × movies matrix of ratings (0 = not rated)
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

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

# Create Flask app
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to Movie Recommender API!', 'status': 'running'})

# API endpoints
@app.route('/api/health', methods=['GET'])

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

if __name__ == '__main__':
    print("🚀 Starting API at http://localhost:8000")
    app.run(debug=True, port=8000)