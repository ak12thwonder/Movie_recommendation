"""
Simple Streamlit UI for Movie Recommendation System
Clean UI that imports functions from existing modules
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import psycopg2
import numpy as np

# Import functions from existing modules
from data_analysis.analysis import load_ratings, load_movies, load_users
from logic.colloborative_filtering import run_recommender
from logic.Calculating_accuracy import evaluate_model
from sklearn.model_selection import train_test_split

# Database config (already present)
db_config = {
    "host": "localhost",
    "dbname": "12thwonder",
    "user": "postgres",
    "password": "admin",
    "port": "5432"
}

# This function
def ensure_recommendation_table(db_config):
    with psycopg2.connect(**db_config) as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS recommendation (
                    user_id INTEGER,
                    movie_recommendate TEXT,
                    PRIMARY KEY (user_id, movie_recommendate)
                )
            """)
        conn.commit()

def save_recommendations(user_id, movie_recommendates, db_config):
    ensure_recommendation_table(db_config)
    with psycopg2.connect(**db_config) as conn:
        with conn.cursor() as cursor:
            for movie_recommendate in movie_recommendates:
                cursor.execute("""
                    INSERT INTO recommendation (user_id, movie_recommendate)
                    VALUES (%s, %s)
                    ON CONFLICT (user_id, movie_recommendate) DO NOTHING
                """, (int(user_id), str(movie_recommendate)))
        conn.commit()




# Page config
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Title
st.title("🎬 Movie Recommendation System")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Dashboard", "Data Analysis", "Recommendations", "Model Evaluation"]
)

# Load data
@st.cache_data
def load_data():
    # ratings = load_ratings("data/ml-100k/u.data")
    # movies = load_movies("data/ml-100k/u.item")
    # users = load_users("data/ml-100k/u.user")
    # return ratings, movies, users

    with psycopg2.connect(**db_config) as conn:
        ratings = pd.read_sql_query("SELECT * FROM ratings", conn)
        movies = pd.read_sql_query("SELECT * FROM movies", conn)
        users = pd.read_sql_query("SELECT * FROM users", conn)
    return ratings, movies, users

st.cache_data.clear()
ratings, movies, users = load_data()

print("** Rating ***",ratings.columns)
print("** movies ***",movies.columns)
print("** users ***",users.columns)


# ============================================================================
# DASHBOARD PAGE
# ============================================================================

if page == "Dashboard":
    st.header("📊 Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", users['user_id'].nunique())
    
    with col2:
        st.metric("Total Movies", movies['movie_id'].nunique())
    
    with col3:
        st.metric("Total Ratings", len(ratings))
    
    with col4:
        st.metric("Avg Rating", f"{ratings['rating'].mean():.2f}")
    
    # Popular movies
    st.subheader("🎬 Most Popular Movies")
    # st.markdown("**X-axis:** Number of ratings (popularity count)  \n**Y-axis:** Movie title")

    popular_movies = ratings.merge(movies[['movie_id', 'title']], on='movie_id')['title'].value_counts().head(10)

    fig = px.bar(
        x=popular_movies.values, 
        y=popular_movies.index, 
        orientation='h',
        title="Top 10 Most Popular Movies",
        labels={"x": "Number of Ratings", "y": "Movie Title"}
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# DATA ANALYSIS PAGE
# ============================================================================

elif page == "Data Analysis":
    st.header("📈 Data Analysis")
    
    # Analysis options
    analysis_type = st.selectbox(
        "Choose analysis:",
        ["Rating Distribution", "Genre Analysis", "User Demographics"]
    )
    
    if analysis_type == "Rating Distribution":
        st.subheader("📊 Rating Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating histogram
            fig = px.histogram(ratings, x='rating', nbins=5, title="Rating Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Rating stats
            st.write("**Rating Statistics:**")
            st.dataframe(ratings['rating'].describe())
    
    elif analysis_type == "Genre Analysis":
        st.subheader("🎭 Genre Analysis")
        
        # Get genre columns
        genre_cols = [col for col in movies.columns 
                     if col not in ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL']]
        
        # Calculate average rating by genre
        genre_data = ratings.merge(movies, on='movie_id')
        genre_ratings = []
        
        for genre in genre_cols:
            genre_movies = genre_data[genre_data[genre] == 1]
            if len(genre_movies) > 0:
                avg_rating = genre_movies['rating'].mean()
                genre_ratings.append({'genre': genre, 'avg_rating': avg_rating})
        
        genre_df = pd.DataFrame(genre_ratings).sort_values('avg_rating', ascending=False)
        
        fig = px.bar(
            genre_df, 
            x='avg_rating', 
            y='genre', 
            orientation='h',
            title="Average Rating by Genre"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "User Demographics":
        st.subheader("👥 User Demographics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            fig_age = px.histogram(users, x='age', nbins=20, title="User Age Distribution")
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # Gender distribution
            gender_counts = users['gender'].value_counts()
            fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index, title="Gender Distribution")
            st.plotly_chart(fig_gender, use_container_width=True)

        # --- Top 5 Movies Loved by Males and Females ---
        # Join ratings, users, and movies
        ratings_users_movies = ratings.merge(users[['user_id', 'gender']], on='user_id').merge(
            movies[['movie_id', 'title']], on='movie_id'
        )

        def get_top_movies_by_gender(df, gender, top_n=5, min_ratings=10):
            gender_df = df[df['gender'] == gender]
            movie_stats = (
                gender_df.groupby('title')['rating']
                .agg(['mean', 'count'])
                .reset_index()
                .rename(columns={'mean': 'avg_rating', 'count': 'num_ratings'})
            )
            # Filter to movies with at least min_ratings
            movie_stats = movie_stats[movie_stats['num_ratings'] >= min_ratings]
            top_movies = movie_stats.sort_values('avg_rating', ascending=False).head(top_n)
            return top_movies

        top_male_movies = get_top_movies_by_gender(ratings_users_movies, 'M', top_n=5)
        top_female_movies = get_top_movies_by_gender(ratings_users_movies, 'F', top_n=5)

        # List of genre columns (update if your schema uses different names)
        genre_columns = [
            'unknown', 'action', 'adventure', 'animation', 'children', 'comedy', 'crime',
            'documentary', 'drama', 'fantasy', 'filmnoir', 'horror', 'musical', 'mystery',
            'romance', 'scifi', 'thriller', 'war', 'western'
        ]

        def get_movie_genres(movie_row):
            return [genre for genre in genre_columns if movie_row.get(genre, 0) == 1]

        def make_movie_genre_table(top_movies, movies_df):
            # Merge to get genre columns for each movie
            merged = top_movies.merge(movies_df[['title'] + genre_columns], on='title', how='left')
            # Build a DataFrame with movie and genres
            data = []
            for _, row in merged.iterrows():
                genres = get_movie_genres(row)
                data.append({'Movie Title': row['title'], 'Genres': ', '.join(genres)})
            return pd.DataFrame(data)

        # Tables for males and females
        male_table = make_movie_genre_table(top_male_movies, movies)
        female_table = make_movie_genre_table(top_female_movies, movies)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Top 5 Movies Loved by Males (with Genres)")
            st.dataframe(male_table, use_container_width=True)

        with col4:
            st.subheader("Top 5 Movies Loved by Females (with Genres)")
            st.dataframe(female_table, use_container_width=True)

# ============================================================================
# RECOMMENDATIONS PAGE
# ============================================================================

elif page == "Recommendations":
    st.header("🎯 Movie Recommendations")
    
    # User selection
    available_users = sorted(ratings['user_id'].unique())
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_id = st.selectbox("Choose a user ID:", available_users, index=99)  # Default to user 100
    
    with col2:
        num_recommendations = st.slider("Number of recommendations:", 1, 20, 5)
    
    # Show user's current ratings
    st.subheader(f"📋 User {user_id}'s Current Ratings")
    
    user_ratings = ratings[ratings['user_id'] == user_id].merge(
        movies[['movie_id', 'title']], on='movie_id'
    ).sort_values('rating', ascending=False)
    
    if len(user_ratings) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Rated Movies:**")
            for _, row in user_ratings.head(5).iterrows():
                st.write(f"⭐ {row['rating']}/5 - {row['title']}")
        
        with col2:
            st.write("**Recently Rated Movies:**")
            for _, row in user_ratings.tail(5).iterrows():
                st.write(f"⭐ {row['rating']}/5 - {row['title']}")
    else:
        st.warning("This user has no ratings yet.")
    
    # Generate Recommendations button
    if st.button("🎯 Generate Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            try:
                recommendations = run_recommender(user_id, num_recommendations)
                st.session_state['last_recommendations'] = recommendations
                st.session_state['last_user_id'] = user_id
                st.success("Recommendations generated! Scroll down to save them.")
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")

    # Display recommendations if present
    if 'last_recommendations' in st.session_state and st.session_state['last_recommendations'] is not None:
        recommendations = st.session_state['last_recommendations']
        display_user_id = st.session_state.get('last_user_id')
        if display_user_id is None:
            display_user_id = user_id
        st.subheader(f"🎬 Recommended Movies for User {display_user_id}")
        for i, (_, row) in enumerate(recommendations.iterrows(), 1):
            st.write(f"**{i}.** {row['title']} (Predicted Rating: {row['predicted_rating']:.2f}/5)")

        if st.button("💾 Save Recommendations"):
            recommended_titles = recommendations['title'].astype(str).tolist()
            try:
                uid = int(display_user_id)
            except (TypeError, ValueError):
                uid = int(user_id)
            save_recommendations(uid, recommended_titles, db_config)
            st.success("Recommendations saved to the database!")

# ============================================================================
# MODEL EVALUATION PAGE
# ============================================================================

elif page == "Model Evaluation":
    st.header("📊 Model Evaluation")
    
    st.subheader("🔬 Collaborative Filtering Performance")
    
    # Parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Test Size:", 0.1, 0.5, 0.2, 0.05)
    
    with col2:
        k_min = st.slider("Min K:", 2, 10, 2)
    
    with col3:
        k_max = st.slider("Max K:", 10, 50, 20)
    
    if st.button("🔍 Find Optimal K (Elbow Method)"):
        with st.spinner("Evaluating K values..."):
            from sklearn.model_selection import train_test_split

            train_df, test_df = train_test_split(ratings, test_size=test_size, random_state=42)
            k_values = list(range(k_min, k_max + 1))
            rmse_scores = []
            for k in k_values:
                try:
                    rmse, mae, count = evaluate_model(train_df, test_df, k=k)
                    rmse_scores.append(rmse)
                except Exception as e:
                    rmse_scores.append(np.nan)  # In case of error, skip this k

            # Find the elbow (minimum RMSE)
            min_rmse = np.nanmin(rmse_scores)
            best_k = k_values[np.nanargmin(rmse_scores)]

            fig = px.line(
                x=k_values, y=rmse_scores,
                labels={"x": "K (Number of Neighbors)", "y": "RMSE"},
                title="Elbow Method for Optimal K"
            )
            fig.add_scatter(x=[best_k], y=[min_rmse], mode='markers+text', text=[f"Best K={best_k}"], textposition="top center", marker=dict(size=12, color='red'), name="Optimal K")
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"Optimal K (elbow point) is {best_k} with RMSE={min_rmse:.4f}")

# Footer
st.markdown("---")
st.markdown("Made by Aditya Kaushal | Movie Recommendation")


