const API_BASE = 'http://localhost:8000';

// ── Film strip ──────────────────────────────────────────────
function fillHoles(containerId) {
    const el = document.getElementById(containerId);
    const count = Math.ceil(window.innerWidth / 28) + 4;
    el.innerHTML = Array(count).fill('<div class="hole"></div>').join('');
}

// ── Tabs ────────────────────────────────────────────────────
function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(btn.dataset.tab).classList.add('active');
        });
    });
}

// ── UI helpers ──────────────────────────────────────────────
function showErr(id, msg) {
    const el = document.getElementById(id);
    el.textContent = msg;
    el.style.display = 'block';
}

function clearErr(id) {
    const el = document.getElementById(id);
    el.textContent = '';
    el.style.display = 'none';
}

function setLoading(id, on) {
    document.getElementById(id).style.display = on ? 'block' : 'none';
}

// ── Results renderer ────────────────────────────────────────
function renderResults(containerId, items) {
    const el = document.getElementById(containerId);
    if (!items || items.length === 0) {
        el.innerHTML = '<p class="msg error" style="display:block">No results found.</p>';
        return;
    }

    const cards = items.map((item, i) => {
        const score = item.similarity_score !== undefined
            ? (item.similarity_score * 100).toFixed(0) + '% genre match'
            : 'predicted ' + item.recommendation_score + ' ★';

        const poster = item.poster
            ? `<img class="card-poster" src="${item.poster}" alt="${item.title}" loading="lazy">`
            : `<div class="card-poster-placeholder">no poster</div>`;

        const overview = item.overview
            ? `<p class="card-overview">${item.overview}</p>`
            : '';

        const tmdbRating = item.tmdb_rating
            ? ` &nbsp;·&nbsp; TMDB ${item.tmdb_rating.toFixed(1)}`
            : '';

        return `
            <div class="card">
                ${poster}
                <div class="card-body">
                    <p class="card-num">${String(i + 1).padStart(2, '0')}</p>
                    <p class="card-title">${item.title}</p>
                    <p class="card-score">${score}${tmdbRating}</p>
                    ${overview}
                </div>
            </div>`;
    }).join('');

    el.innerHTML = `<p class="results-heading">Results</p><div class="cards">${cards}</div>`;
}

// ── API calls ───────────────────────────────────────────────
async function fetchSemanticSearch() {
    clearErr('err-search');
    document.getElementById('results-search').innerHTML = '';

    const query = document.getElementById('search-query').value.trim();
    const n = document.getElementById('n-search').value;
    if (!query) { showErr('err-search', 'Please describe what you\'re looking for.'); return; }

    setLoading('load-search', true);
    try {
        const res = await fetch(`${API_BASE}/api/search?q=${encodeURIComponent(query)}&n=${n}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Search failed.');
        renderResults('results-search', data.recommendations);
    } catch (e) {
        showErr('err-search', e.message);
    } finally {
        setLoading('load-search', false);
    }
}

async function fetchSimilarMovies() {
    clearErr('err-movie');
    document.getElementById('results-movie').innerHTML = '';

    const title = document.getElementById('movie-title').value.trim();
    const n = document.getElementById('n-movie').value;
    if (!title) { showErr('err-movie', 'Please enter a film title.'); return; }

    setLoading('load-movie', true);
    try {
        const res = await fetch(`${API_BASE}/api/movies/${encodeURIComponent(title)}?n=${n}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Film not found in catalogue.');
        renderResults('results-movie', data.recommendations);
    } catch (e) {
        showErr('err-movie', e.message);
    } finally {
        setLoading('load-movie', false);
    }
}

async function fetchUserRecommendations() {
    clearErr('err-user');
    document.getElementById('results-user').innerHTML = '';

    const userId = document.getElementById('user-id').value.trim();
    const n = document.getElementById('n-user').value;
    if (!userId) { showErr('err-user', 'Please enter a viewer ID.'); return; }

    setLoading('load-user', true);
    try {
        const res = await fetch(`${API_BASE}/api/recommend/user/${userId}?n=${n}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Viewer not found.');
        renderResults('results-user', data.recommendations);
    } catch (e) {
        showErr('err-user', e.message);
    } finally {
        setLoading('load-user', false);
    }
}

// ── Enter key shortcuts ─────────────────────────────────────
function initEnterKeys() {
    ['search-query', 'n-search'].forEach(id => {
        document.getElementById(id).addEventListener('keydown', e => {
            if (e.key === 'Enter') fetchSemanticSearch();
        });
    });
    ['movie-title', 'n-movie'].forEach(id => {
        document.getElementById(id).addEventListener('keydown', e => {
            if (e.key === 'Enter') fetchSimilarMovies();
        });
    });
    ['user-id', 'n-user'].forEach(id => {
        document.getElementById(id).addEventListener('keydown', e => {
            if (e.key === 'Enter') fetchUserRecommendations();
        });
    });
}

// ── Init ────────────────────────────────────────────────────
fillHoles('holes-top');
fillHoles('holes-bottom');
initTabs();
initEnterKeys();

document.getElementById('btn-search').addEventListener('click', fetchSemanticSearch);
document.getElementById('btn-movie').addEventListener('click', fetchSimilarMovies);
document.getElementById('btn-user').addEventListener('click', fetchUserRecommendations);
