"""
Simple Streamlit UI for Movie Recommendation System
Clean UI that imports functions from existing modules
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import psycopg2

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
    ratings = load_ratings("data/ml-100k/u.data")
    movies = load_movies("data/ml-100k/u.item")
    users = load_users("data/ml-100k/u.user")
    return ratings, movies, users

ratings, movies, users = load_data()

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
    popular_movies = ratings.merge(movies[['movie_id', 'title']], on='movie_id')['title'].value_counts().head(10)
    
    fig = px.bar(
        x=popular_movies.values, 
        y=popular_movies.index, 
        orientation='h',
        title="Top 10 Most Popular Movies"
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
    
    # Generate recommendations button
    if st.button("🎯 Generate Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            try:
                recommendations = run_recommender(user_id, num_recommendations)
                
                # Save recommendations to PostgreSQL
                # recommended_movie_recommendates = recommendations['movie_recommendate'].tolist()
                recommended_movie_recommendates = recommendations['title'].tolist()
                save_recommendations(user_id, recommended_movie_recommendates, db_config)
                
                st.subheader(f"🎬 Recommended Movies for User {user_id}")
                
                for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                    st.write(f"**{i}.** {row['title']} (Predicted Rating: {row['predicted_rating']:.2f}/5)")
                
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")

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
        k_neighbors = st.slider("K Neighbors:", 5, 20, 10)
    
    with col3:
        random_state = st.number_input("Random State:", value=42, min_value=1, max_value=1000)
    
    # Run evaluation button
    if st.button("🚀 Run Evaluation", type="primary"):
        with st.spinner("Evaluating model performance..."):
            try:
                # Split data
                train_df, test_df = train_test_split(ratings, test_size=test_size, random_state=random_state)
                
                # Evaluate model
                rmse, mae, count = evaluate_model(train_df, test_df, k=k_neighbors)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("RMSE", f"{rmse:.4f}")
                
                with col2:
                    st.metric("MAE", f"{mae:.4f}")
                
                with col3:
                    st.metric("Test Samples", f"{count:,}")
                
                # Performance interpretation
                st.subheader("📈 Performance Analysis")
                
                if rmse < 1.0:
                    st.success("✅ Excellent performance! RMSE below 1.0 indicates very good predictions.")
                elif rmse < 1.5:
                    st.info("ℹ️ Good performance! RMSE between 1.0-1.5 indicates reasonable predictions.")
                else:
                    st.warning("⚠️ Performance could be improved. Consider tuning hyperparameters.")
                
                # Show sample data
                st.subheader("🔍 Sample Test Data")
                st.dataframe(test_df[['user_id', 'movie_id', 'rating']].head(10))
                
            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Made with Aditya Kaushal | Movie Recommendation")


