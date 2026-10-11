import streamlit as st
import pickle
import pandas as pd
import requests


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



def show_poster(url):
    if url:
        st.image(url, use_container_width=True)
    else:
        st.text("Poster unavailable")




@st.cache_data
def load_movies():
    with open("movies.pkl", "rb") as f:
        movies_data = pickle.load(f)

    return pd.DataFrame(movies_data)


@st.cache_data
def load_similarity():
    with open("similirety_top10.pkl", "rb") as f:
        return pickle.load(f)


movies = load_movies()
similirety = load_similarity()



def recommend(movie):
    matched_movies = movies[movies["title"] == movie]

    if matched_movies.empty:
        return [], []

    movie_index = matched_movies.index[0]

    distances = similirety.get(movie_index, [])

    movie_list = distances[:5]

    recommended = []
    movie_poster = []

    for index, score in movie_list:
        movie_title = movies.iloc[index]["title"]

        recommended.append(movie_title)
        movie_poster.append(fetch_poster(movie_title))

    return recommended, movie_poster



st.title("Movie Recommendation System")

selected_movies_name = st.selectbox(
    "Top 5 Movies on Your Interests",
    movies["title"].values
)

if st.button("Recommend"):
    names, posters = recommend(selected_movies_name)

    if names:
        columns = st.columns(5)

        for i, col in enumerate(columns):
            with col:
                if i < len(names):
                    st.text(names[i])
                    show_poster(posters[i])
    else:
        st.warning("No recommendations found for this movie.")



st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: chocolate;
    }

    [data-testid="stHeader"] {
        background-color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.text("Search Over 5000+ Movies")
