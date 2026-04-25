#!/usr/bin/env python3
"""
Run this script once locally before deploying to precompute genome and
collaborative-filtering factors from the large CSV files that are excluded
from git.

Output files (committed to .cache/):
  genome_factors.npy   — 13k × 100 compressed genome tag vectors
  genome_mask.npy      — boolean mask: which movies have genome data
  cf_item_factors.npy  — 13k × 100 SVD item factors from ratings

Usage:
  python precompute.py
  git add .cache/genome_factors.npy .cache/genome_mask.npy .cache/cf_item_factors.npy
"""
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, '.cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Load movies ──────────────────────────────────────────────
movies = pd.read_csv(os.path.join(BASE_DIR, 'ml-25m/movies_filtered.csv'))
n_movies = len(movies)
movie_id_to_idx = {mid: i for i, mid in enumerate(movies['movieId'].values)}
movie_ids = set(movies['movieId'].values)
print(f"Loaded {n_movies} movies.")

# ── 1. Genome factors ────────────────────────────────────────
genome_path = os.path.join(BASE_DIR, 'ml-25m/genome-scores.csv')
if os.path.exists(genome_path):
    print("\n── Genome tags ─────────────────────────────────────")
    print("Loading genome-scores.csv...")
    genome_scores = pd.read_csv(genome_path)
    genome_scores = genome_scores[genome_scores['movieId'].isin(movie_ids)]

    genome_pivot = genome_scores.pivot(
        index='movieId', columns='tagId', values='relevance'
    ).fillna(0)

    n_tags = genome_pivot.shape[1]
    genome_raw = np.zeros((n_movies, n_tags), dtype='float32')
    genome_mask = np.zeros(n_movies, dtype=bool)

    for movie_id in genome_pivot.index:
        if movie_id in movie_id_to_idx:
            idx = movie_id_to_idx[movie_id]
            genome_raw[idx] = genome_pivot.loc[movie_id].values.astype('float32')
            genome_mask[idx] = True

    print(f"{genome_mask.sum()} / {n_movies} movies have genome data.")

    print("Compressing with SVD (100 components)...")
    svd = TruncatedSVD(n_components=100, random_state=42)
    genome_factors = svd.fit_transform(genome_raw).astype('float32')

    np.save(os.path.join(CACHE_DIR, 'genome_factors.npy'), genome_factors)
    np.save(os.path.join(CACHE_DIR, 'genome_mask.npy'), genome_mask)
    print(f"Saved genome_factors.npy {genome_factors.shape} and genome_mask.npy")
else:
    print("genome-scores.csv not found — skipping genome factors.")

# ── 2. CF item factors ───────────────────────────────────────
ratings_path = os.path.join(BASE_DIR, 'ml-25m/ratings_filtered.csv')
if os.path.exists(ratings_path):
    print("\n── Collaborative filtering ──────────────────────────")
    print("Loading ratings_filtered.csv...")
    ratings = pd.read_csv(ratings_path, usecols=['userId', 'movieId', 'rating'])
    ratings = ratings[ratings['movieId'].isin(movie_ids)]

    user_ids = ratings['userId'].unique()
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    n_users = len(user_ids)

    ratings['user_idx'] = ratings['userId'].map(user_id_to_idx)
    ratings['movie_idx'] = ratings['movieId'].map(movie_id_to_idx)
    ratings = ratings.dropna(subset=['movie_idx'])
    ratings['movie_idx'] = ratings['movie_idx'].astype(int)

    sparse = csr_matrix(
        (ratings['rating'].values.astype('float32'),
         (ratings['user_idx'].values, ratings['movie_idx'].values)),
        shape=(n_users, n_movies)
    )
    print(f"Sparse matrix: {sparse.shape}, {sparse.nnz:,} ratings.")

    print("Running SVD (100 components) — this may take 1-2 minutes...")
    svd = TruncatedSVD(n_components=100, random_state=42)
    svd.fit(sparse)
    item_factors = svd.components_.T.astype('float32')  # n_movies × 100

    np.save(os.path.join(CACHE_DIR, 'cf_item_factors.npy'), item_factors)
    print(f"Saved cf_item_factors.npy {item_factors.shape}")
else:
    print("ratings_filtered.csv not found — skipping CF factors.")

print("\n── Done ────────────────────────────────────────────────")
print("Now run:")
print("  git add .cache/genome_factors.npy .cache/genome_mask.npy .cache/cf_item_factors.npy")
print("  git commit -m 'Add precomputed genome and CF factors'")
print("  git push")
