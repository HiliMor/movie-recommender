from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

os.chdir('/Users/hilimor/movie-recommender')

# Load data
ratings = pd.read_csv('ml-100k/u.data', sep='\t', header=None,
                      names=['userId', 'movieId', 'rating', 'timestamp'])

movies = pd.read_csv('ml-100k/u.item', sep='|', header=None, encoding='latin-1',
                     names=['movieId', 'title', 'releaseDate', 'videoReleaseDate', 'url', 
                            'unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy', 
                            'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror', 
                            'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western'])

# Build matrices
genre_columns = ['unknown', 'action', 'adventure', 'animation', 'childrens', 'comedy', 
                 'crime', 'documentary', 'drama', 'fantasy', 'film_noir', 'horror', 
                 'musical', 'mystery', 'romance', 'sci_fi', 'thriller', 'war', 'western']

genre_matrix = movies[genre_columns].values
similarity_matrix = cosine_similarity(genre_matrix)

user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
user_similarity = cosine_similarity(user_movie_matrix)

# Recommender functions
def recommend_movies_improved(movie_title, n_recommendations=5):
    try:
        movie_idx = movies[movies['title'] == movie_title].index[0]
    except IndexError:
        return None

    similarity_scores = similarity_matrix[movie_idx]
    similar_movie_indices = similarity_scores.argsort()[-n_recommendations-2:-2][::-1]

    return [{'title': movies.iloc[idx]['title'], 'similarity_score': float(similarity_scores[idx])} 
            for idx in similar_movie_indices]

def recommend_movies_collaborative(user_id, n_recommendations=5):
    if user_id not in ratings['userId'].values:
        return None

    user_idx = user_id - 1
    similarity_scores = user_similarity[user_idx]
    similar_user_indices = similarity_scores.argsort()[-11:-1][::-1]
    similar_users = [idx + 1 for idx in similar_user_indices]

    movies_rated_by_similar = ratings[(ratings['userId'].isin(similar_users)) & (ratings['rating'] >= 4)].copy()
    user_rated_movies = set(ratings[ratings['userId'] == user_id]['movieId'].values)
    recommended_movie_ids = movies_rated_by_similar[~movies_rated_by_similar['movieId'].isin(user_rated_movies)]['movieId'].values

    movie_counts = {}
    for movie_id in recommended_movie_ids:
        movie_counts[movie_id] = movie_counts.get(movie_id, 0) + 1

    top_movies = sorted(movie_counts.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]

    return [{'title': movies[movies['movieId'] == movie_id]['title'].values[0], 'recommendation_score': int(count)} 
            for movie_id, count in top_movies] if top_movies else None

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
    recs = recommend_movies_improved(movie_title, n_recommendations=n)
    if recs is None:
        return jsonify({'error': f'Movie not found: {movie_title}'}), 404
    return jsonify({'movie': movie_title, 'recommendations': recs})

@app.route('/api/recommend/user/<int:user_id>', methods=['GET'])
def recommend_user(user_id):
    n = request.args.get('n', 5, type=int)
    recs = recommend_movies_collaborative(user_id, n_recommendations=n)
    if recs is None:
        return jsonify({'error': f'User not found: {user_id}'}), 404
    return jsonify({'user_id': user_id, 'recommendations': recs})

if __name__ == '__main__':
    print("🚀 Starting API at http://localhost:8000")
    app.run(debug=True, port=8000)