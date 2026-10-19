import html
import os
import math
import random
from concurrent.futures import ThreadPoolExecutor
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
# Key kept as before so posters keep working. Set OMDB_API_KEY on Render to override it.
OMDB_KEY = os.environ.get("OMDB_API_KEY", "bfdb5c3a")


def fetch_poster(movie_title):
    try:
        response = requests.get(
            "https://www.omdbapi.com/",
            params={
                "t": movie_title,
                "apikey": OMDB_KEY
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
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; padding-top: 1.2rem; border-radius: 14px; }
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
/* live poster wall behind everything */
.wall-frame {
    position: fixed; inset: 0; width: 100vw; height: 100vh; height: 100dvh;
    overflow: hidden; z-index: 0; pointer-events: none;
}
.wall {
    position: absolute; inset: -12% -8%; pointer-events: none;
    display: flex; flex-direction: column; gap: 18px; justify-content: center;
    transform: rotate(-7deg) scale(1.1);
    opacity: 0.16;
    -webkit-mask-image: radial-gradient(ellipse at 50% 40%, rgba(0,0,0,0.55) 0%, #000 75%);
    mask-image: radial-gradient(ellipse at 50% 40%, rgba(0,0,0,0.55) 0%, #000 75%);
}
.wall .strip { display: flex; gap: 18px; width: max-content; animation: drift 90s linear infinite; }
.wall .strip.back { animation-direction: reverse; animation-duration: 110s; }
.wall .strip.slow { animation-duration: 130s; }
.wall img { height: 210px; width: 140px; object-fit: cover; border-radius: 10px; flex: none; }
@keyframes drift { from { transform: translateX(0); } to { transform: translateX(-50%); } }
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"] { overflow-x: hidden !important; max-width: 100%; }
div[data-testid="stElementContainer"]:has(.wall-frame) { height: 0; margin: 0; padding: 0; }
.block-container {
    position: relative; z-index: 1;
    background: rgba(24, 8, 17, 0.72);
    -webkit-backdrop-filter: blur(6px); backdrop-filter: blur(6px);
    border-radius: 22px;
    padding-left: 2.2rem !important; padding-right: 2.2rem !important;
}
.hero h1, .because b { text-shadow: 0 2px 18px rgba(0, 0, 0, 0.7); }
button[data-baseweb="tab"] p { color: var(--muted); font-weight: 700; }
button[data-baseweb="tab"][aria-selected="true"] p { color: var(--cream); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }
[data-testid="stRadio"] label p { color: var(--cream) !important; }
[data-testid="stPopoverBody"] { background: #2a1220; border: 1px solid rgba(219, 165, 74, 0.5); }
[data-testid="stPopoverBody"] p, [data-testid="stPopoverBody"] li { color: var(--cream) !important; }
.facts { color: var(--cream); line-height: 1.6; font-size: 0.95rem; }
.facts b { color: var(--brass); }
.clue { background: rgba(244, 234, 211, 0.08); border-left: 3px solid var(--brass); padding: 0.7rem 1rem; margin: 0.5rem 0; border-radius: 6px; color: var(--cream); }
.clue b { color: var(--brass); }
@media (max-width: 640px) { .wall img { height: 150px; width: 100px; } .wall { opacity: 0.24; } }
@media (prefers-reduced-motion: reduce) { .wall .strip { animation: none; } }

.ticket .tag {
    position: absolute; top: 10px; right: 10px; z-index: 2;
    background: var(--brass); color: var(--ink); font-weight: 800; font-size: 0.72rem;
    padding: 2px 9px; border-radius: 999px;
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
        ("Trailer", f"https://www.youtube.com/results?search_query={quote_plus(title + ' official trailer')}"),
    ]
    links = "".join(
        f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{name}</a>'
        for name, url in sites
    )
    return f'<div class="watch"><span>Watch on</span>{links}</div>'


def ticket_html(rank, title, poster, note=""):
    safe_title = html.escape(title)
    if poster:
        art = f'<img src="{html.escape(poster, quote=True)}" alt="Poster of {safe_title}" loading="lazy">'
    else:
        art = '<div class="blank"><span>🎬</span>Poster unavailable</div>'
    return (
        f'<div class="ticket" style="--i:{rank - 1}">'
        f'<div class="no">No. {rank}</div>'
        + (f'<div class="tag">{html.escape(note)}</div>' if note else "")
        + f'{art}'
        f'<div class="stub"><div class="name">{safe_title}</div>{watch_links(title)}</div>'
        f'</div>'
    )


WALL_TITLES = [
    "Avatar", "The Dark Knight", "Inception", "Interstellar", "Titanic", "The Avengers",
    "Jurassic Park", "The Matrix", "Gladiator", "Toy Story", "Finding Nemo", "The Lion King",
    "Pulp Fiction", "Forrest Gump", "Fight Club", "The Godfather", "Star Wars", "Skyfall",
    "Iron Man", "Batman Begins", "Shrek", "Up", "Django Unchained", "Mad Max: Fury Road",
    "The Prestige", "Casino Royale", "Spider-Man", "Gravity", "Deadpool", "Inside Out",
    "Kung Fu Panda", "Back to the Future", "Alien", "Blade Runner", "Braveheart", "The Departed",
]


@st.cache_data(show_spinner=False, ttl=86400)
def wall_posters():
    """Fetch posters of well-known movies once (in parallel) for the animated background."""
    titles = [t for t in WALL_TITLES if t in set(movies["title"])]
    with ThreadPoolExecutor(max_workers=8) as pool:
        found = list(pool.map(fetch_poster, titles))
    return [p for p in found if p]


def wall_html(urls):
    if len(urls) < 6:
        return ""
    urls = list(urls)
    random.Random(7).shuffle(urls)
    rows = [urls[0::3], urls[1::3], urls[2::3]]
    classes = ["", "back", "slow"]
    out = []
    for row, cls in zip(rows, classes):
        imgs = "".join(
            f'<img src="{html.escape(u, quote=True)}" alt="" loading="lazy" decoding="async">'
            for u in row + row
        )
        out.append(f'<div class="strip {cls}">{imgs}</div>')
    return '<div class="wall-frame" aria-hidden="true"><div class="wall">' + "".join(out) + "</div></div>"


st.markdown(STYLE, unsafe_allow_html=True)
st.markdown(wall_html(wall_posters()), unsafe_allow_html=True)

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
    # Share link: ?movie=Avatar opens the app with that movie already chosen
    shared = st.query_params.get("movie")
    if shared and shared in set(movies["title"]):
        st.session_state["pick"] = shared

if "favs" not in st.session_state:
    st.session_state["favs"] = []


def toggle_fav(title):
    favs = st.session_state["favs"]
    if title in favs:
        favs.remove(title)
    else:
        favs.append(title)


@st.cache_data(show_spinner=False, ttl=86400)
def omdb_details(title):
    """Year, genre, director, cast, plot and rating from OMDb; None if unavailable."""
    try:
        r = requests.get(
            "https://www.omdbapi.com/",
            params={"t": title, "apikey": OMDB_KEY, "plot": "short"},
            timeout=10,
        )
        r.raise_for_status()
        d = r.json()
        return d if d.get("Response") == "True" else None
    except (requests.exceptions.RequestException, ValueError, AttributeError):
        return None


def card_extras(prefix, i, title):
    """Details popover and save button under a card."""
    with st.popover("Details", width="stretch"):
        d = omdb_details(title)
        if d:
            rows = [
                ("Year", d.get("Year")), ("Genre", d.get("Genre")), ("Director", d.get("Director")),
                ("Cast", d.get("Actors")), ("IMDb rating", d.get("imdbRating")), ("Runtime", d.get("Runtime")),
            ]
            facts = "".join(
                f"<div><b>{k}:</b> {html.escape(str(v))}</div>" for k, v in rows if v and v != "N/A"
            )
            plot = d.get("Plot")
            if plot and plot != "N/A":
                facts += f"<p style='margin-top:0.6rem'>{html.escape(plot)}</p>"
            st.markdown(f'<div class="facts">{facts}</div>', unsafe_allow_html=True)
        else:
            st.write("Details are not available right now.")
    saved = title in st.session_state["favs"]
    st.button(
        "♥ Saved" if saved else "♡ Save",
        key=f"fav_{prefix}_{i}",
        on_click=toggle_fav,
        args=(title,),
        width="stretch",
    )


def set_pick(title):
    st.session_state["pick"] = title


def surprise():
    st.session_state["pick"] = str(movies["title"].sample(1).iloc[0])


@st.cache_data(show_spinner=False)
def cached_recommend(title):
    # same recommend() as above; cached so clicking around does not re-fetch posters
    return recommend(title)


tab_similar, tab_plot, tab_duo, tab_list, tab_game, tab_how = st.tabs(
    ["Similar movies", "Describe the plot", "Movie night for two", "My list", "Guess the movie", "How it works"]
)

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
        st.query_params["movie"] = selected_movies_name
    elif "movie" in st.query_params:
        del st.query_params["movie"]

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
                            width="stretch",
                        )
                        card_extras("sim", i, names[i])
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
                        card_extras("plot", i, found[i])
        else:
            st.warning("No movie matches those words. Try simpler ones, such as war, love, alien or crime.")
    else:
        st.markdown(
            '<div class="hint">Type a few words about the plot, setting or theme, or press one of the ideas above. '
            'We look for them in each movie\'s overview, keywords, genres, cast and director.</div>',
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# Tab 3: movie night for two - one list that suits both people
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def duo_picks(a, b):
    """Combine the neighbour lists of two movies. A movie that both lists contain ranks first."""
    def neighbours(title):
        rows = movies[movies["title"] == title]
        if rows.empty:
            return {}
        out = {}
        for i, sc in similirety.get(rows.index[0], []):
            t = movies.iloc[i]["title"]
            if t not in (a, b):
                out[t] = max(out.get(t, 0), float(sc))
        return out

    na, nb = neighbours(a), neighbours(b)
    both = {t: na[t] + nb[t] for t in na if t in nb}
    ranked = sorted(both, key=both.get, reverse=True)
    singles = {t: sc for d in (na, nb) for t, sc in d.items() if t not in both}
    ranked += sorted(singles, key=singles.get, reverse=True)
    return [(t, "Both" if t in both else "") for t in ranked[:5]]


with tab_duo:
    st.markdown(
        '<div class="because">Two people, one movie<b>Pick one film each</b></div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        duo_a = st.selectbox("First person likes", movies["title"].values, index=None,
                             key="duo_a", placeholder="Type a movie...")
    with c2:
        duo_b = st.selectbox("Second person likes", movies["title"].values, index=None,
                             key="duo_b", placeholder="Type a movie...")

    if duo_a and duo_b and duo_a != duo_b:
        picks = duo_picks(duo_a, duo_b)
        if picks:
            st.caption("Movies marked Both appear in the similar-movie lists of both picks.")
            cols = st.columns(5, gap="medium")
            for i, col in enumerate(cols):
                with col:
                    if i < len(picks):
                        t, note = picks[i]
                        st.markdown(
                            ticket_html(i + 1, t, cached_poster(t), note),
                            unsafe_allow_html=True,
                        )
                        card_extras("duo", i, t)
        else:
            st.warning("No shared picks found for these two movies.")
    elif duo_a and duo_b:
        st.warning("Choose two different movies.")
    else:
        st.markdown(
            '<div class="hint">Choose one movie each. We combine both similar-movie lists '
            'and put movies that suit both of you first.</div>',
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# Tab 4: saved movies
# ----------------------------------------------------------------------------
with tab_list:
    favs = st.session_state["favs"]
    if not favs:
        st.markdown(
            '<div class="hint">Nothing saved yet. Press Save under any movie and it appears here. '
            'The list stays while this page is open.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="because">Your list<b>{len(favs)} saved movie{"s" if len(favs) != 1 else ""}</b></div>',
            unsafe_allow_html=True,
        )
        for start in range(0, len(favs), 5):
            chunk = favs[start:start + 5]
            cols = st.columns(5, gap="medium")
            for j, col in enumerate(cols):
                with col:
                    if j < len(chunk):
                        t = chunk[j]
                        st.markdown(
                            ticket_html(start + j + 1, t, cached_poster(t)),
                            unsafe_allow_html=True,
                        )
                        card_extras("list", start + j, t)

# ----------------------------------------------------------------------------
# Tab 5: guess the movie
# ----------------------------------------------------------------------------
def mask_title(text, title):
    for word in re.findall(r"[A-Za-z0-9']+", title):
        if len(word) > 3:
            text = re.sub(re.escape(word), "_____", text, flags=re.I)
    return text


def new_round():
    pool = [t for t in WALL_TITLES if t in set(movies["title"])]
    random.shuffle(pool)
    for answer in pool:
        d = omdb_details(answer)
        if d:
            others = random.sample([t for t in pool if t != answer], 3)
            options = others + [answer]
            random.shuffle(options)
            st.session_state["round"] = {"answer": answer, "options": options, "d": d}
            st.session_state["checked"] = False
            st.session_state.pop("guess", None)
            return
    st.session_state["round"] = None


if "score" not in st.session_state:
    st.session_state["score"] = [0, 0]


def check_round():
    st.session_state["checked"] = True
    r = st.session_state.get("round")
    if r and st.session_state.get("guess") is not None:
        st.session_state["score"][1] += 1
        if st.session_state["guess"] == r["answer"]:
            st.session_state["score"][0] += 1


with tab_game:
    if "round" not in st.session_state:
        new_round()
    r = st.session_state["round"]
    if not r:
        st.warning("Clues could not be loaded right now. Check the internet connection and try again.")
        st.button("Try again", on_click=new_round, key="retry_round")
    else:
        d = r["d"]
        right, total = st.session_state["score"]
        st.markdown(
            f'<div class="because">Guess the movie<b>Score {right} of {total}</b></div>',
            unsafe_allow_html=True,
        )
        clues = [
            ("Year", d.get("Year")), ("Genre", d.get("Genre")),
            ("Director", d.get("Director")), ("Cast", d.get("Actors")),
        ]
        for k, v in clues:
            if v and v != "N/A":
                st.markdown(f'<div class="clue"><b>{k}:</b> {html.escape(str(v))}</div>', unsafe_allow_html=True)
        plot = d.get("Plot")
        if plot and plot != "N/A":
            st.markdown(
                f'<div class="clue"><b>Plot:</b> {html.escape(mask_title(plot, r["answer"]))}</div>',
                unsafe_allow_html=True,
            )
        st.radio("Which movie is it?", r["options"], index=None, key="guess",
                 disabled=st.session_state.get("checked", False))
        b1, b2 = st.columns(2)
        with b1:
            st.button("Check answer", on_click=check_round, key="check_btn",
                      disabled=st.session_state.get("checked", False), width="stretch")
        with b2:
            st.button("Next movie", on_click=new_round, key="next_btn", width="stretch")
        if st.session_state.get("checked"):
            if st.session_state.get("guess") == r["answer"]:
                st.success(f"Correct. It is {r['answer']}.")
            elif st.session_state.get("guess") is None:
                st.info(f"No answer chosen. It was {r['answer']}.")
            else:
                st.error(f"Not this one. It is {r['answer']}.")

# ----------------------------------------------------------------------------
# Tab 6: how it works
# ----------------------------------------------------------------------------
with tab_how:
    steps = [
        ("Collect", "Every movie gets one block of text: overview, keywords, genres, top three cast and the director."),
        ("Clean", "Words are lower-cased and reduced to their stem, so run, runs and running count as one word."),
        ("Count", f"The 5,000 most common words become the columns. Each of the {n_titles:,} movies becomes a row of word counts."),
        ("Compare", "Cosine similarity measures how close two rows are. Movies with similar words get a high score."),
        ("Store", "For each movie only the 10 closest movies are saved. This keeps the file small enough for a free server."),
        ("Show", "The app reads the list for your movie, skips repeats and shows five, with posters from OMDb."),
    ]
    for n, (head, body) in enumerate(steps, 1):
        st.markdown(
            f'<div class="clue"><b>{n}. {head}:</b> {html.escape(body)}</div>',
            unsafe_allow_html=True,
        )
    st.caption("This is content-based filtering. It looks only at what a movie is about, not at what other people watched.")
