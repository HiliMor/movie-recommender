# Picture House — Movie Recommender

A movie recommendation system built from scratch using the MovieLens 25M dataset. Deployed at [movie-recommendations.up.railway.app](https://movie-recommendations.up.railway.app).

---

## Features

### Search — describe what you want
Type a description in plain English: *"funny film for a rainy evening"* or *"intense spy thriller"*. No need to know a specific title.

Uses **TF-IDF** with synonym expansion to match your description against movie titles and genres. Results are blended with a popularity signal (65% relevance, 35% popularity) so well-known films surface ahead of obscure ones for broad queries.

### Similar films — start from a title you know
Enter a film you enjoyed and the engine finds the most similar films using three signals blended together:

| Signal | Weight | What it captures |
|---|---|---|
| **Genome tags** | 50% | 1,128 crowd-sourced descriptors per film (*"feel-good"*, *"thought-provoking"*, *"plot twist"*) compressed to 100 dimensions with SVD |
| **Collaborative filtering** | 35% | Films that the same audiences rate similarly — SVD on a sparse 160k users × 13k movies ratings matrix |
| **Genre overlap** | 15% | Binary genre vector (Action, Comedy, Drama…) as a tie-breaker |

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| Data | Pandas, NumPy, SciPy |
| ML | scikit-learn — TF-IDF, TruncatedSVD, cosine similarity |
| API | Flask, Flask-CORS, Gunicorn |
| Movie enrichment | TMDB API (posters, overviews, ratings) |
| Frontend | HTML, CSS, vanilla JS |
| Deployment | Railway |

---

## Project structure

```
movie-recommender/
├── app.py               # Flask API and recommendation logic
├── precompute.py        # One-time script to build genome + CF factors
├── index.html           # Frontend
├── about.html           # How it works page
├── style.css
├── main.js
├── requirements.txt
├── ml-25m/              # MovieLens 25M dataset (not in git — large files)
│   └── movies_filtered.csv   # committed — filtered movie list
├── .cache/              # Precomputed numpy arrays (committed)
│   ├── popularity.npy
│   ├── genome_factors.npy
│   ├── genome_mask.npy
│   └── cf_item_factors.npy
└── .env                 # TMDB API token (not in git)
```

---

## Local setup

### 1. Get the dataset

Download [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) and place the contents in `ml-25m/`.

### 2. Get a TMDB API token

Create a free account at [themoviedb.org](https://www.themoviedb.org), go to Settings → API, and copy the **API Read Access Token**.

Create a `.env` file:
```
TMDB_TOKEN=your_token_here
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Precompute factors (first time only)

```bash
python precompute.py
```

This reads `genome-scores.csv` and `ratings_filtered.csv` (large files, not in git) and saves small precomputed numpy arrays to `.cache/`. Takes about 2 minutes.

### 5. Run

```bash
python app.py
```

Open `http://localhost:8000` — do not open `index.html` directly as a file.

---

## API

```
GET /api/search?q=funny+spy+thriller&n=5
GET /api/movies/Toy%20Story%20(1995)?n=5
GET /api/movies/autocomplete?q=toy
GET /api/health
```

All recommendation responses include `title`, `similarity_score`, `genres`, `poster`, `overview`, and `tmdb_rating`.

---

## Data

**MovieLens 25M** — 25 million ratings from 162,000 users on 62,000 movies. Filtered to movies with 50+ ratings → 13,176 movies.

**MovieLens Genome** — 1,128 crowd-sourced relevance tags per film, scored 0–1. Compressed from 1,128 → 100 dimensions with SVD before use.

---

## What I learned building this

**Recommendation techniques**
- Content-based filtering: representing movies as genre vectors and finding nearest neighbours with cosine similarity
- Collaborative filtering: SVD on a sparse user–item matrix to find latent taste dimensions — the same principle behind Netflix and Spotify
- Hybrid recommenders: combining multiple signals with weighted blending
- Why genre-only similarity is limited and how genome tags and CF complement it

**Working with data**
- Building sparse matrices from ratings data with SciPy
- Dimensionality reduction with Truncated SVD
- TF-IDF vectorisation and query expansion for text search

**Engineering**
- Keeping memory low enough for a free-tier deployment (replaced PyTorch sentence-transformers with TF-IDF, compute similarity on-demand instead of precomputing N×N matrices)
- Pre-computing expensive factors locally and committing the small output files
- Flask serving both API and static frontend from a single service
- Deploying on Railway with Gunicorn
