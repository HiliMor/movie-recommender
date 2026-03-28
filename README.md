# Movie Recommender System

A machine learning recommendation system built from scratch as a learning project. Covers content-based filtering, collaborative filtering with SVD, semantic search with HuggingFace embeddings, and a TMDB-enriched frontend.

## How it works

### 1. Similar films (content-based)

Given a movie title, finds films with overlapping genres using cosine similarity on genre vectors.

Each movie is represented as a binary vector across 19 genre flags. Cosine similarity measures the angle between two vectors — small angle means similar genres, regardless of how many genres each film has.

### 2. For a viewer (collaborative filtering — SVD)

Given a user ID, predicts ratings for every movie the user hasn't seen and returns the highest-predicted ones.

Uses **Truncated SVD** (matrix factorization) on the 943×1682 user-movie ratings matrix. SVD decomposes the matrix into latent factors — hidden "taste dimensions" like affinity for slow dramas or action films — without you ever labeling them. Reconstructing the matrix fills in the blanks with predicted ratings.

This replaces the earlier nearest-neighbor approach, which only looked at who rated what. SVD considers how much users rated things and finds deeper patterns across all 100,000 ratings at once.

### 3. Search (semantic search — HuggingFace)

Takes a plain English description like *"dark psychological thriller"* or *"funny film for kids"* and returns the closest matching movies.

Uses `all-MiniLM-L6-v2` from HuggingFace, a sentence transformer model trained on over 1 billion text pairs. It maps any text to a 384-dimensional vector (an "embedding") that encodes meaning. At startup, all 1682 movies are embedded once. At query time, the query is embedded with the same model, and cosine similarity finds the nearest movies in that 384-dimensional space.

Because it measures angle (not distance), a short query like *"war drama"* and a long one like *"an intense emotional film set during a conflict with a focus on human suffering"* land in roughly the same place.

### 4. TMDB enrichment

Every recommendation is enriched with poster, overview, and rating from the TMDB API. Movie titles are parsed from the MovieLens format (`"Star Wars (1977)"`) into title + year, searched against TMDB, and the first result is used.

---

## Project structure

```
movie-recommender/
├── app.py                        # Flask API — models, endpoints
├── index.html                    # Frontend
├── style.css                     # Styles
├── main.js                       # Frontend logic
├── 01_load_explore_data.ipynb    # Data exploration notebook
├── ml-100k/                      # MovieLens 100K dataset (not in git)
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

**MovieLens 100K** — 100,000 ratings from 943 users on 1,682 movies, collected 1987–1998. Ratings on a 1–5 scale, 19 genre categories.

Small by modern standards, but ideal for learning — fast to iterate, clean, and the standard benchmark used in ML literature.

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
