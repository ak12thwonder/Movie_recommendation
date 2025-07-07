# app.py

import streamlit as st
import pandas as pd
import psycopg2

from data.load_movielens import load_ratings, load_movies
from logic.collab_filtering import get_user_ratings  # function to be defined
# from logic.db_utils import ensure_recommendation_table, save_recommendations

st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎥 Movie Recommender System")
st.subheader("🔎 Explore Your Ratings")

# Load ratings & movies data (use caching to speed up)
@st.cache_data
def load_data():
    ratings = load_ratings()
    movies = load_movies()
    return ratings, movies

ratings_df, movies_df = load_data()

# Step 1️⃣: User Input
user_id = st.number_input("Enter User ID:", min_value=1, max_value=943, step=1)

# Step 2️⃣: Show Past Ratings
if user_id:
    user_ratings = get_user_ratings(user_id, ratings_df, movies_df)

    if not user_ratings.empty:
        st.success(f"Showing past ratings for User {user_id}")
        st.dataframe(user_ratings[['title', 'rating']].sort_values(by='rating', ascending=False),
                     use_container_width=True)
    else:
        st.warning("This user has no ratings.")

def ensure_recommendation_table(db_config):
    with psycopg2.connect(**db_config) as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recommendation (
                    user_id INTEGER,
                    movie_id INTEGER,
                    PRIMARY KEY (user_id, movie_id)
                )
            """)
        conn.commit()

def save_recommendations(user_id, movie_ids, db_config):
    ensure_recommendation_table(db_config)
    with psycopg2.connect(**db_config) as conn:
        with conn.cursor() as cursor:
            for movie_id in movie_ids:
                cursor.execute("""
                    INSERT INTO recommendation (user_id, movie_id)
                    VALUES (%s, %s)
                    ON CONFLICT (user_id, movie_id) DO NOTHING
                """, (user_id, movie_id))
        conn.commit()
