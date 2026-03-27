# 🎬 Movie Recommender System

A complete machine learning recommendation system built from scratch using content-based filtering, collaborative filtering, and hybrid approaches.

## Features

### 1. Content-Based Recommendations

Find movies similar to a movie you like based on genres

- Uses cosine similarity between movie genre vectors
- Fast and interpretable

### 2. Collaborative Filtering

Get personalized recommendations based on user ratings

- Finds users with similar rating patterns
- Recommends movies highly-rated by similar users
- Learns from user preferences

### 3. Hybrid Approach

Combines both methods for best results

- Balances genre similarity with user preferences
- More accurate recommendations

## Project Structure

```
movie-recommender/
├── 01_load_explore_data.ipynb    # Data loading and EDA
├── app.py                         # Flask API server
├── ml-100k/                       # MovieLens dataset
├── venv/                          # Python virtual environment
└── README.md                      # This file
```

## Installation & Setup

### 1. Clone/Download the project

```bash
cd movie-recommender
```

### 2. Activate virtual environment

```bash
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate     # Windows
```

### 3. Install dependencies (if needed)

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter flask flask-cors
```

### 4. Run the API

```bash
python app.py
```

You should see:

```
🚀 Starting API at http://localhost:8000
 * Debugger is active!
```

## API Usage

### Health Check

```
GET http://localhost:8000/api/health
```

Response:

```json
{
  "status": "healthy",
  "movies": 1682,
  "users": 943
}
```

### Get Similar Movies (Content-Based)

```
GET http://localhost:8000/api/movies/Star%20Wars%20%281977%29?n=5
```

Parameters:

- `movie_title` (URL): Movie name (URL encoded)
- `n` (query): Number of recommendations (default: 5)

Response:

```json
{
  "movie": "Star Wars (1977)",
  "recommendations": [
    {
      "title": "Empire Strikes Back, The (1980)",
      "similarity_score": 0.913
    },
    ...
  ]
}
```

### Get User Recommendations (Collaborative Filtering)

```
GET http://localhost:8000/api/recommend/user/1?n=5
```

Parameters:

- `user_id` (URL): User ID (1-943)
- `n` (query): Number of recommendations (default: 5)

Response:

```json
{
  "user_id": 1,
  "recommendations": [
    {
      "title": "Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1963)",
      "recommendation_score": 8
    },
    ...
  ]
}
```

## Data

Uses the **MovieLens 100K Dataset**:

- 100,000 ratings from 943 users on 1,682 movies
- Ratings on a 1-5 scale
- 19 movie genres

## Technologies Used

- **Python**: Core language
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning (cosine similarity)
- **Flask**: REST API framework
- **Jupyter**: Data exploration

## Results

### Content-Based Performance

- Finds movies with similar genres
- Fast and explainable
- Works well for similar movie discovery

### Collaborative Filtering Performance

- Personalized to user preferences
- Learns from rating patterns
- Better for discovering new genres users might like

### Example Results

**For Star Wars fans:**

- Recommends: Empire Strikes Back, Star Trek, Starship Troopers
- Reason: Similar sci-fi/adventure genres

**For User 1 (mixed taste):**

- Recommends: Dr. Strangelove, Schindler's List, One Flew Over the Cuckoo's Nest
- Reason: Other users with similar taste liked these

## Next Steps & Improvements

### Short-term

- [ ] Add web UI (HTML/CSS/JavaScript)
- [ ] Add movie search functionality
- [ ] Add user registration and ratings
- [ ] Improve similarity scoring with weighted genres

### Medium-term

- [ ] Deploy to cloud (Heroku, AWS, Google Cloud)
- [ ] Add database (SQLite, PostgreSQL)
- [ ] Implement matrix factorization (SVD)
- [ ] Add deep learning models (neural networks)

### Long-term

- [ ] Real-time rating updates
- [ ] Cold-start problem solutions
- [ ] A/B testing framework
- [ ] Advanced metrics (precision@K, NDCG, MAP)

## Skills Learned

✅ Data loading and exploration
✅ Similarity metrics (cosine similarity)
✅ Content-based filtering
✅ Collaborative filtering
✅ Matrix operations and linear algebra
✅ REST API design
✅ Flask web framework
✅ Machine learning workflows

## License

MIT License - Feel free to use for learning and projects

## Author

Built as a learning project for understanding recommendation systems from first principles.

---

## Running the Jupyter Notebook

To explore the data and see the analysis:

```bash
jupyter notebook 01_load_explore_data.ipynb
```

This notebook contains:

- Data loading and exploration
- Similarity matrix calculations
- Content-based recommender
- Collaborative filtering recommender
- Hybrid recommender
- Visualizations and statistics
