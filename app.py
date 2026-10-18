import html
import math
import pickle
import re
from urllib.parse import quote_plus

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
)


# ----------------------------------------------------------------------------
# Data and recommendation logic (same as before; see the two marked fixes below)
# ----------------------------------------------------------------------------
def fetch_poster(movie_title):
    try:
        response = requests.get(
            "https://www.omdbapi.com/",
            params={
                "t": movie_title,
                "apikey": "bfdb5c3a"
            },
            timeout=10
        )

        response.raise_for_status()
        data = response.json()

        poster = data.get("Poster")

        if isinstance(poster, str) and poster.startswith("http"):
            return poster

        return None

    except (
        requests.exceptions.RequestException,
        ValueError,
        AttributeError
    ):
        return None


@st.cache_data
def load_movies():
    with open("movies.pkl", "rb") as f:
        movies_data = pickle.load(f)

    # FIX 1: row labels must equal row positions (0..N-1), because the similarity
    # file is keyed by position. Without this, movies after row 2658 got the
    # neighbours of a different movie.
    return pd.DataFrame(movies_data).reset_index(drop=True)


@st.cache_data
def load_similarity():
    with open("similirety_top10.pkl", "rb") as f:
        return pickle.load(f)


STOP = {"the", "and", "for", "with", "that", "who", "movie", "film", "about", "from", "into", "his", "her"}

movies = load_movies()
similirety = load_similarity()


def recommend(movie):
    matched_movies = movies[movies["title"] == movie]

    if matched_movies.empty:
        return [], []

    movie_index = matched_movies.index[0]

    distances = similirety.get(movie_index, [])

    # FIX 2: some titles occur several times in the data. Skip the selected title
    # itself and titles that were already chosen, then take five.
    movie_list = []
    seen_titles = {movie}
    for i, s in distances:
        t = movies.iloc[i]["title"]
        if t in seen_titles:
            continue
        seen_titles.add(t)
        movie_list.append((i, s))
        if len(movie_list) == 5:
            break

    recommended = []
    movie_poster = []

    for index, score in movie_list:
        movie_title = movies.iloc[index]["title"]

        recommended.append(movie_title)
        movie_poster.append(fetch_poster(movie_title))

    return recommended, movie_poster


# ----------------------------------------------------------------------------
# Look and feel
# ----------------------------------------------------------------------------
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700;9..144,900&family=Manrope:wght@400;500;700&display=swap');

:root {
    --velvet: #1b0b14;
    --velvet-2: #3a1226;
    --brass: #dba54a;
    --brass-dark: #b5822c;
    --cream: #f4ead3;
    --ink: #241018;
    --muted: #c9b8a4;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1100px 520px at 12% -8%, var(--velvet-2) 0%, transparent 70%),
        radial-gradient(900px 480px at 100% 0%, #2c0f1d 0%, transparent 65%),
        var(--velvet);
    color: var(--cream);
    font-family: 'Manrope', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
.block-container { max-width: 1180px; padding-top: 2.4rem; padding-bottom: 4rem; }

/* headline */
.hero { margin: 0.5rem 0 2rem 0; max-width: 760px; }
.hero h1 {
    font-family: 'Fraunces', serif;
    font-weight: 900;
    font-size: clamp(2.4rem, 6vw, 4.2rem);
    line-height: 1.02;
    letter-spacing: -0.02em;
    color: var(--cream);
    margin: 0 0 0.9rem 0;
    padding: 0;
}
.hero p {
    font-size: 1.08rem;
    line-height: 1.6;
    color: var(--muted);
    margin: 0;
    max-width: 52ch;
}
.hero .count { color: var(--brass); font-weight: 700; }

/* picker */
[data-testid="stWidgetLabel"] p {
    color: var(--cream) !important;
    font-weight: 700;
    font-size: 0.95rem;
}
[data-baseweb="select"] > div {
    background: rgba(244, 234, 211, 0.07);
    border: 1px solid rgba(219, 165, 74, 0.45);
    border-radius: 10px;
    min-height: 3rem;
    color: var(--cream);
}
[data-baseweb="select"] > div:hover { border-color: var(--brass); }
[data-baseweb="select"] input, [data-baseweb="select"] div { color: var(--cream); }
/* suggestion list: always on top, readable, clearly separate from the page */
[data-baseweb="popover"], [data-baseweb="popover"] > div { z-index: 999999 !important; }
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] ul {
    background: #2a1220 !important;
    border: 1px solid rgba(219, 165, 74, 0.6);
    border-radius: 10px;
    box-shadow: 0 24px 40px -10px rgba(0, 0, 0, 0.8);
    max-height: 360px;
}
[data-baseweb="popover"] [role="option"] {
    background: transparent !important;
    color: var(--cream) !important;
    padding: 0.55rem 0.9rem;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [role="option"][aria-selected="true"] {
    background: rgba(219, 165, 74, 0.22) !important;
}

div.stButton > button {
    background: var(--brass);
    color: var(--ink);
    border: 0;
    border-radius: 10px;
    min-height: 3rem;
    padding: 0 1.6rem;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 1rem;
    width: 100%;
    transition: background 0.15s ease, transform 0.15s ease;
}
div.stButton > button:hover { background: #ecb95d; color: var(--ink); transform: translateY(-1px); }
div.stButton > button:active { transform: translateY(1px); }
div.stButton > button:focus-visible { outline: 3px solid var(--cream); outline-offset: 2px; }

/* results */
.because { margin: 2.2rem 0 1.1rem 0; color: var(--muted); font-size: 1rem; }
.because b {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 1.5rem;
    color: var(--cream);
    display: block;
    margin-top: 0.15rem;
}
.ticket {
    position: relative;
    background: var(--cream);
    color: var(--ink);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 18px 30px -14px rgba(0, 0, 0, 0.7);
    animation: ticket-in 0.55s cubic-bezier(0.2, 0.7, 0.2, 1) both;
    animation-delay: calc(var(--i) * 90ms);
}
.ticket .no {
    position: absolute; top: 10px; left: 10px; z-index: 2;
    background: var(--ink); color: var(--brass);
    font-family: 'Fraunces', serif; font-weight: 700; font-size: 0.95rem;
    padding: 2px 10px; border-radius: 999px;
}
.ticket img, .ticket .blank {
    display: block; width: 100%; aspect-ratio: 2 / 3; object-fit: cover;
}
.ticket .blank {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 0.6rem; text-align: center; padding: 1rem;
    background: linear-gradient(160deg, #4a1a31, #1b0b14);
    color: var(--muted); font-size: 0.85rem;
}
.ticket .blank span { font-size: 2.2rem; }
.ticket .stub {
    position: relative; padding: 1rem 0.9rem 1.05rem 0.9rem;
    border-top: 2px dashed rgba(36, 16, 24, 0.28);
}
.ticket .stub::before, .ticket .stub::after {
    content: ""; position: absolute; top: -11px; width: 20px; height: 20px;
    border-radius: 50%; background: var(--velvet);
}
.ticket .stub::before { left: -10px; }
.ticket .stub::after { right: -10px; }
.ticket .name {
    font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.02rem;
    line-height: 1.25; min-height: 2.6em;
}
.ticket .watch { display: flex; flex-wrap: wrap; gap: 0.35rem; align-items: center; margin-top: 0.7rem; }
.ticket .watch span { font-size: 0.78rem; color: #6b5560; width: 100%; }
.ticket .watch a {
    font-size: 0.82rem; font-weight: 700; text-decoration: none;
    color: var(--cream); background: var(--ink);
    padding: 0.3rem 0.65rem; border-radius: 999px;
    transition: background 0.15s ease;
}
.ticket .watch a:hover { background: var(--velvet-2); }
.ticket .watch a:focus-visible { outline: 3px solid var(--brass-dark); outline-offset: 2px; }
@keyframes ticket-in {
    from { opacity: 0; transform: translateY(18px) rotate(-1deg); }
    to { opacity: 1; transform: none; }
}
@media (prefers-reduced-motion: reduce) {
    .ticket { animation: none; }
    div.stButton > button { transition: none; }
    .ticket .watch a { transition: none; }
}

/* "more like this" buttons under each card */
div[data-testid="stColumn"] div.stButton > button.more,
div[data-testid="stColumn"] div.stButton > button {
    min-height: 2.4rem;
}
/* phones: Streamlit stacks the columns, so show each card as a compact row */
@media (max-width: 640px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1.2rem; }
    .hero { margin-bottom: 1.2rem; }
    .hero p { font-size: 1rem; }
    .because b { font-size: 1.25rem; }
    .ticket { display: grid; grid-template-columns: 96px 1fr; align-items: stretch; border-radius: 12px; }
    .ticket img, .ticket .blank { aspect-ratio: 2 / 3; height: 100%; }
    .ticket .blank { font-size: 0.7rem; padding: 0.4rem; }
    .ticket .blank span { font-size: 1.4rem; }
    .ticket .no { top: 6px; left: 6px; font-size: 0.8rem; padding: 1px 8px; }
    .ticket .stub { border-top: 0; border-left: 2px dashed rgba(36, 16, 24, 0.28); display: flex; align-items: center; }
    .ticket .stub::before, .ticket .stub::after { display: none; }
    .ticket .name { min-height: 0; font-size: 1.05rem; }
    .ticket .stub { flex-direction: column; align-items: flex-start; justify-content: center; }
    .ticket .watch a { padding: 0.4rem 0.75rem; }
    div[data-testid="stColumn"] div.stButton > button { min-height: 2.75rem; }
}
.hint {
    margin-top: 1.6rem; padding: 1.1rem 1.3rem; max-width: 640px;
    border-left: 3px solid var(--brass); color: var(--muted); line-height: 1.6;
}
</style>
"""


def watch_links(title):
    """Search links on streaming sites. They open a search for the title; availability depends on the viewer's country."""
    q = quote_plus(title)
    sites = [
        ("Netflix", f"https://www.netflix.com/search?q={q}"),
        ("Prime Video", f"https://www.primevideo.com/search/?phrase={q}"),
        ("JustWatch", f"https://www.justwatch.com/in/search?q={q}"),
    ]
    links = "".join(
        f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{name}</a>'
        for name, url in sites
    )
    return f'<div class="watch"><span>Watch on</span>{links}</div>'


def ticket_html(rank, title, poster):
    safe_title = html.escape(title)
    if poster:
        art = f'<img src="{html.escape(poster, quote=True)}" alt="Poster of {safe_title}" loading="lazy">'
    else:
        art = '<div class="blank"><span>🎬</span>Poster unavailable</div>'
    return (
        f'<div class="ticket" style="--i:{rank - 1}">'
        f'<div class="no">No. {rank}</div>{art}'
        f'<div class="stub"><div class="name">{safe_title}</div>{watch_links(title)}</div>'
        f'</div>'
    )


st.markdown(STYLE, unsafe_allow_html=True)

n_titles = movies["title"].nunique()
st.markdown(
    f"""
    <div class="hero">
        <h1>Find your next movie.</h1>
        <p>Choose a film you like and we will list five that read alike, from plot, cast,
        director and genre. Search across <span class="count">{n_titles:,}</span> movies.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "pick" not in st.session_state:
    st.session_state["pick"] = None


def set_pick(title):
    st.session_state["pick"] = title


def surprise():
    st.session_state["pick"] = str(movies["title"].sample(1).iloc[0])


@st.cache_data(show_spinner=False)
def cached_recommend(title):
    # same recommend() as above; cached so clicking around does not re-fetch posters
    return recommend(title)


tab_similar, tab_plot = st.tabs(["Similar movies", "Describe the plot"])

with tab_similar:
    pick_col, rand_col = st.columns([4, 1], vertical_alignment="bottom")
    with pick_col:
        selected_movies_name = st.selectbox(
            "Search for a movie you like",
            movies["title"].values,
            index=None,
            key="pick",
            placeholder="Start typing, for example: Avatar, Dark Knight, Toy Story...",
        )
    with rand_col:
        st.button("Surprise me", on_click=surprise)

    if selected_movies_name:
        with st.spinner("Finding similar movies..."):
            names, posters = cached_recommend(selected_movies_name)

        if names:
            st.markdown(
                f'<div class="because">Because you picked<b>{html.escape(selected_movies_name)}</b></div>',
                unsafe_allow_html=True,
            )
            columns = st.columns(5, gap="medium")

            for i, col in enumerate(columns):
                with col:
                    if i < len(names):
                        st.markdown(
                            ticket_html(i + 1, names[i], posters[i]),
                            unsafe_allow_html=True,
                        )
                        st.button(
                            "More like this",
                            key=f"more_{i}",
                            on_click=set_pick,
                            args=(names[i],),
                        )
        else:
            st.warning("No recommendations found for this movie.")
    else:
        st.markdown(
            '<div class="hint">Type a title in the box above, or press Surprise me. '
            'Five similar movies appear here with their posters, and you can press '
            'More like this on any of them to keep exploring.</div>',
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# Tab 2: find movies by describing the story in a few words
# ----------------------------------------------------------------------------
IDEAS = ["space war", "heist", "haunted house", "time travel", "road trip", "revenge", "robot", "friendship"]


def set_idea(text):
    st.session_state["plot_q"] = text


@st.cache_data(show_spinner=False)
def cached_poster(title):
    return fetch_poster(title)


@st.cache_data(show_spinner=False)
def search_by_words(query):
    """Rank movies by how many of the typed words appear in their tags (rarer words count more)."""
    words = [w for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 2 and w not in STOP]
    if not words:
        return [], []
    tags = movies["tags"].astype(str)
    score = pd.Series(0.0, index=movies.index)
    used = []
    for w in dict.fromkeys(words):
        stem = w[: max(min(len(w), 5), len(w) - 3)]
        hit = tags.str.contains(r"\b" + re.escape(stem), regex=True)
        df = int(hit.sum())
        if df == 0:
            continue
        used.append(w)
        score += hit * math.log(len(movies) / (1 + df))
    order = score[score > 0].sort_values(ascending=False)
    result, seen = [], set()
    for idx in order.index:
        t = movies.iloc[idx]["title"]
        if t in seen:
            continue
        seen.add(t)
        result.append(t)
        if len(result) == 5:
            break
    return result, used


with tab_plot:
    st.text_input(
        "Describe the kind of story you want",
        key="plot_q",
        placeholder="For example: space war, heist, haunted house...",
    )
    chip_cols = st.columns(4)
    for n, idea in enumerate(IDEAS):
        with chip_cols[n % 4]:
            st.button(idea, key=f"idea_{n}", on_click=set_idea, args=(idea,))

    query = st.session_state.get("plot_q", "").strip()
    if query:
        found, used = search_by_words(query)
        if found:
            st.markdown(
                f'<div class="because">Best matches for<b>{html.escape(", ".join(used))}</b></div>',
                unsafe_allow_html=True,
            )
            cols = st.columns(5, gap="medium")
            for i, col in enumerate(cols):
                with col:
                    if i < len(found):
                        st.markdown(
                            ticket_html(i + 1, found[i], cached_poster(found[i])),
                            unsafe_allow_html=True,
                        )
        else:
            st.warning("No movie matches those words. Try simpler ones, such as war, love, alien or crime.")
    else:
        st.markdown(
            '<div class="hint">Type a few words about the plot, setting or theme, or press one of the ideas above. '
            'We look for them in each movie\'s overview, keywords, genres, cast and director.</div>',
            unsafe_allow_html=True,
        )
