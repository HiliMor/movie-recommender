# Movie Recommender System

A machine learning recommendation system built from scratch as a learning project. Covers content-based filtering, collaborative filtering with SVD, semantic search with HuggingFace embeddings, and a TMDB-enriched frontend.

## The three ways to find a film

### Search — describe what you want

Type a description in plain English, like *"funny film for a rainy evening"* or *"intense spy thriller"*. No need to know a specific title.

Under the hood it uses a sentence embedding model (`all-MiniLM-L6-v2`) to turn your description and every film in the catalogue into a numeric vector that encodes meaning. The closest matches are returned. Results are blended with a popularity signal so well-known films surface ahead of obscure ones for broad queries.

### Similar Films — start from a title you know

Already enjoyed a film and want more like it? Type its title (autocomplete will help), and the engine finds movies that share the same genre profile.

Each film is represented as a binary vector of 19 genre flags (Action, Comedy, Drama…). Similarity is measured by the angle between vectors — films with heavily overlapping genres come out on top.

### For a Viewer — personalised picks by user ID

Enter a viewer ID from the MovieLens dataset to get recommendations tailored to that person's taste. The engine looks at the ratings of thousands of similar viewers and predicts which unseen films that user would rate most highly.

This uses SVD (matrix factorisation) on a 2,560 active users × 13,000 movies ratings matrix. SVD finds hidden "taste dimensions" — patterns like *affinity for slow dramas* or *preference for action* — without them ever being labelled. Reconstructing the matrix fills in predicted ratings for every film a user hasn't seen.

### TMDB enrichment

Every result is enriched with a poster, overview, and rating from the TMDB API. Movie titles are parsed from the MovieLens format (`"Star Wars (1977)"`) into title + year, then looked up on TMDB.

---

## Project structure

```
movie-recommender/
├── app.py                        # Flask API — models, endpoints
├── index.html                    # Frontend
├── style.css                     # Styles
├── main.js                       # Frontend logic
├── 01_load_explore_data.ipynb    # ML-100K data exploration
├── 02_explore_25m.ipynb          # ML-25M exploration and filtering
├── ml-25m/                       # MovieLens 25M dataset (not in git)
├── .env                          # TMDB API token (not in git)
├── venv/                         # Python virtual environment (not in git)
└── README.md
```

---

## Setup

### 1. Get the dataset

Download [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) and place the `ml-100k/` folder in the project root.

### 2. Get a TMDB API token

Create a free account at [themoviedb.org](https://www.themoviedb.org), go to Settings → API, and copy the **API Read Access Token**.

Create a `.env` file:

```
TMDB_TOKEN=your_token_here
```

### 3. Install dependencies

```bash
source venv/bin/activate
pip install pandas numpy scikit-learn flask flask-cors requests python-dotenv sentence-transformers torch transformers jupyter
```

### 4. Run the API

```bash
python app.py
```

The server starts at `http://localhost:8000`. Open `index.html` directly in your browser.

---

## API endpoints

### Health check
```
GET /api/health
```

### Similar films
```
GET /api/movies/Toy%20Story%20(1995)?n=5
```

### User recommendations
```
GET /api/recommend/user/42?n=5
```

### Semantic search
```
GET /api/search?q=funny+movie+for+kids&n=5
```

All recommendation responses include `title`, a score field, `poster`, `overview`, and `tmdb_rating`.

---

## Data

**MovieLens 25M** — 25 million ratings from 162,000 users on 62,000 movies, collected up to 2019. Ratings on a 0.5–5 scale, 19 genre categories.

For performance, the app filters to:
- Movies with 50+ ratings — 13,176 movies
- Users with 1,000+ ratings for SVD — 2,560 most active users

This keeps the recommendation quality high (more signal, less noise) while keeping the SVD matrix at a manageable size (~33M cells vs 1.3B unfiltered).

The full exploration and filtering process is documented in `02_explore_25m.ipynb`.

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Data | Pandas, NumPy |
| ML | Scikit-learn (SVD, cosine similarity) |
| Embeddings | sentence-transformers, HuggingFace `all-MiniLM-L6-v2` |
| API | Flask, Flask-CORS |
| Movie data | TMDB API |
| Frontend | HTML, CSS, vanilla JS |
| Notebook | Jupyter |

---

## What I learned building this

This project was built step by step, each piece introducing a new concept.

**Machine learning concepts**
- What cosine similarity actually measures — angle between vectors, not distance — and why that matters for text
- Content-based filtering: representing movies as genre vectors and finding nearest neighbors
- Collaborative filtering: why nearest-neighbor is limited and how SVD fixes it by finding latent patterns across the entire dataset
- Matrix factorization: decomposing a users × movies matrix into hidden "taste dimensions"
- Embeddings: how a neural network maps text into a vector space where meaning determines position
- The difference between genre-based similarity (what a movie *is*) and semantic similarity (what a movie *feels like*)

**Working with data**
- Loading and exploring datasets with Pandas
- Building and manipulating matrices with NumPy
- What the MovieLens dataset looks like and how ratings data is structured

**APIs and external services**
- How REST APIs work — making HTTP requests, reading JSON responses
- Authentication with Bearer tokens vs API keys
- Integrating the TMDB API to enrich recommendations with posters and descriptions
- Loading a pre-trained model from HuggingFace in one line and using it immediately

**Software structure**
- Separating a project into HTML / CSS / JS — one responsibility per file
- Building a Flask backend and connecting it to a frontend via fetch calls
- Keeping secrets out of code with `.env` files and `.gitignore`
